// SM89 FP8 W8A8 CUTLASS backend.
//
// This mirrors vLLM's main idea for Ada FP8 GEMM: dispatch different CUTLASS
// tile families by the runtime M bucket instead of using one generic kernel.

#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <torch/extension.h>

#include "cutlass/cutlass.h"
#include "cutlass/arch/mma.h"
#include "cutlass/array.h"
#include "cutlass/float8.h"
#include "cutlass/functional.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/numeric_conversion.h"

namespace {

using ElementA = cutlass::float_e4m3_t;
using ElementB = cutlass::float_e4m3_t;
using ElementC = cutlass::bfloat16_t;
using ElementAccumulator = float;
using ElementCompute = float;
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 32>;

constexpr int kAlignmentA = 16;
constexpr int kAlignmentB = 16;

template <
    typename ElementOutput_,
    int Count,
    typename ElementAccumulator_ = ElementOutput_,
    typename ElementCompute_ = ElementOutput_>
class W8A8DeviceScaleCombination {
 public:
  using ElementOutput = ElementOutput_;
  using ElementAccumulator = ElementAccumulator_;
  using ElementCompute = ElementCompute_;

  static int const kCount = Count;
  using FragmentOutput = cutlass::Array<ElementOutput, kCount>;
  using FragmentAccumulator = cutlass::Array<ElementAccumulator, kCount>;
  using FragmentSource = cutlass::Array<ElementOutput, kCount>;
  using FragmentCompute = cutlass::Array<ElementCompute, kCount>;

  struct Params {
    ElementCompute alpha;
    ElementCompute beta;
    ElementCompute const* x_scale_ptr;
    ElementCompute const* w_scale_ptr;

    CUTLASS_HOST_DEVICE
    Params()
        : alpha(ElementCompute(1)),
          beta(ElementCompute(0)),
          x_scale_ptr(nullptr),
          w_scale_ptr(nullptr) {}

    CUTLASS_HOST_DEVICE
    Params(ElementCompute const* x_scale_ptr, ElementCompute const* w_scale_ptr)
        : alpha(ElementCompute(1)),
          beta(ElementCompute(0)),
          x_scale_ptr(x_scale_ptr),
          w_scale_ptr(w_scale_ptr) {}
  };

 private:
  Params params_;

 public:
  CUTLASS_HOST_DEVICE
  W8A8DeviceScaleCombination(Params const& params) : params_(params) {
    if (params_.x_scale_ptr && params_.w_scale_ptr) {
      ElementCompute x_scale = *params_.x_scale_ptr;
      x_scale = x_scale < ElementCompute(1.0e-12f) ? ElementCompute(1.0e-12f) : x_scale;
      params_.alpha = x_scale * (*params_.w_scale_ptr);
    }
  }

  CUTLASS_HOST_DEVICE
  bool is_source_needed() const {
    return false;
  }

  CUTLASS_HOST_DEVICE
  void set_k_partition(int k_partition, int k_partition_count) {
    if (k_partition) {
      params_.beta = ElementCompute(1);
    }
  }

  CUTLASS_HOST_DEVICE
  FragmentOutput operator()(FragmentAccumulator const& accumulator, FragmentOutput const& source) const {
    cutlass::NumericArrayConverter<ElementCompute, ElementOutput, kCount> source_converter;
    cutlass::NumericArrayConverter<ElementCompute, ElementAccumulator, kCount> accumulator_converter;
    cutlass::NumericArrayConverter<ElementOutput, ElementCompute, kCount> destination_converter;

    FragmentCompute converted_source = source_converter(source);
    FragmentCompute converted_accumulator = accumulator_converter(accumulator);
    FragmentCompute intermediate;
    cutlass::multiply_add<FragmentCompute> mul_add_accumulator;

    intermediate = mul_add_accumulator(params_.alpha, converted_accumulator, params_.beta * converted_source);
    return destination_converter(intermediate);
  }

  CUTLASS_HOST_DEVICE
  FragmentOutput operator()(FragmentAccumulator const& accumulator) const {
    cutlass::NumericArrayConverter<ElementCompute, ElementAccumulator, kCount> accumulator_converter;
    cutlass::NumericArrayConverter<ElementOutput, ElementCompute, kCount> destination_converter;

    FragmentCompute converted_accumulator = accumulator_converter(accumulator);
    cutlass::multiplies<FragmentCompute> mul_accumulator;
    return destination_converter(mul_accumulator(params_.alpha, converted_accumulator));
  }
};

template <typename TileShape, typename WarpShape, int Stages, typename MathOperator>
using W8A8Gemm = cutlass::gemm::device::Gemm<
    ElementA,
    LayoutA,
    ElementB,
    LayoutB,
    ElementC,
    LayoutC,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm89,
    TileShape,
    WarpShape,
    InstructionShape,
    W8A8DeviceScaleCombination<
        ElementC,
        128 / cutlass::sizeof_bits<ElementC>::value,
        ElementAccumulator,
        ElementCompute>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    Stages,
    kAlignmentA,
    kAlignmentB,
    false,
    MathOperator>;

uint32_t next_pow2(uint32_t value) {
  if (value <= 1) {
    return 1;
  }
  --value;
  value |= value >> 1;
  value |= value >> 2;
  value |= value >> 4;
  value |= value >> 8;
  value |= value >> 16;
  return value + 1;
}

template <typename Gemm>
void run_cutlass_gemm(
    const torch::Tensor& a,
    const torch::Tensor& b,
    torch::Tensor& out,
    const torch::Tensor& x_scale,
    const torch::Tensor& w_scale) {
  const int m = static_cast<int>(a.size(0));
  const int k = static_cast<int>(a.size(1));
  const int n = static_cast<int>(b.size(1));

  typename Gemm::Arguments args(
      {m, n, k},
      {reinterpret_cast<ElementA const*>(a.data_ptr()), static_cast<int>(a.stride(0))},
      {reinterpret_cast<ElementB const*>(b.data_ptr()), static_cast<int>(b.stride(1))},
      {reinterpret_cast<ElementC const*>(out.data_ptr()), static_cast<int>(out.stride(0))},
      {reinterpret_cast<ElementC*>(out.data_ptr()), static_cast<int>(out.stride(0))},
      {x_scale.data_ptr<float>(), w_scale.data_ptr<float>()});

  Gemm gemm;
  cutlass::Status status = gemm.can_implement(args);
  TORCH_CHECK(status == cutlass::Status::kSuccess, "CUTLASS W8A8 can_implement failed");

  size_t workspace_size = Gemm::get_workspace_size(args);
  torch::Tensor workspace;
  void* workspace_ptr = nullptr;
  if (workspace_size != 0) {
    workspace = torch::empty(
        {static_cast<long>(workspace_size)},
        torch::TensorOptions().dtype(torch::kUInt8).device(a.device()));
    workspace_ptr = workspace.data_ptr();
  }

  status = gemm.initialize(args, workspace_ptr, at::cuda::getCurrentCUDAStream());
  TORCH_CHECK(status == cutlass::Status::kSuccess, "CUTLASS W8A8 initialize failed");
  status = gemm(at::cuda::getCurrentCUDAStream());
  TORCH_CHECK(status == cutlass::Status::kSuccess, "CUTLASS W8A8 run failed");
}

template <typename OutType>
void dispatch_default(
    const torch::Tensor& a,
    const torch::Tensor& b,
    torch::Tensor& out,
    const torch::Tensor& x_scale,
    const torch::Tensor& w_scale) {
  (void)sizeof(OutType);
  const uint32_t n_pow2 = next_pow2(static_cast<uint32_t>(out.size(1)));
  if (n_pow2 <= 4096) {
    using Gemm = W8A8Gemm<
        cutlass::gemm::GemmShape<128, 128, 64>,
        cutlass::gemm::GemmShape<64, 64, 64>,
        5,
        cutlass::arch::OpMultiplyAddFastAccum>;
    run_cutlass_gemm<Gemm>(a, b, out, x_scale, w_scale);
  } else if (n_pow2 <= 8192) {
    using Gemm = W8A8Gemm<
        cutlass::gemm::GemmShape<256, 128, 64>,
        cutlass::gemm::GemmShape<64, 64, 64>,
        3,
        cutlass::arch::OpMultiplyAddFastAccum>;
    run_cutlass_gemm<Gemm>(a, b, out, x_scale, w_scale);
  } else {
    using Gemm = W8A8Gemm<
        cutlass::gemm::GemmShape<128, 128, 64>,
        cutlass::gemm::GemmShape<64, 64, 64>,
        5,
        cutlass::arch::OpMultiplyAddFastAccum>;
    run_cutlass_gemm<Gemm>(a, b, out, x_scale, w_scale);
  }
}

void dispatch_m256(
    const torch::Tensor& a,
    const torch::Tensor& b,
    torch::Tensor& out,
    const torch::Tensor& x_scale,
    const torch::Tensor& w_scale) {
  const uint32_t n_pow2 = next_pow2(static_cast<uint32_t>(out.size(1)));
  if (n_pow2 <= 4096) {
    using Gemm = W8A8Gemm<
        cutlass::gemm::GemmShape<64, 128, 128>,
        cutlass::gemm::GemmShape<64, 64, 64>,
        3,
        cutlass::arch::OpMultiplyAddFastAccum>;
    run_cutlass_gemm<Gemm>(a, b, out, x_scale, w_scale);
  } else {
    using Gemm = W8A8Gemm<
        cutlass::gemm::GemmShape<128, 128, 64>,
        cutlass::gemm::GemmShape<64, 64, 64>,
        5,
        cutlass::arch::OpMultiplyAddFastAccum>;
    run_cutlass_gemm<Gemm>(a, b, out, x_scale, w_scale);
  }
}

void dispatch_m128(
    const torch::Tensor& a,
    const torch::Tensor& b,
    torch::Tensor& out,
    const torch::Tensor& x_scale,
    const torch::Tensor& w_scale) {
  const uint32_t n_pow2 = next_pow2(static_cast<uint32_t>(out.size(1)));
  if (n_pow2 <= 8192) {
    using Gemm = W8A8Gemm<
        cutlass::gemm::GemmShape<64, 128, 128>,
        cutlass::gemm::GemmShape<64, 64, 64>,
        3,
        cutlass::arch::OpMultiplyAddFastAccum>;
    run_cutlass_gemm<Gemm>(a, b, out, x_scale, w_scale);
  } else if (n_pow2 <= 16384) {
    using Gemm = W8A8Gemm<
        cutlass::gemm::GemmShape<128, 128, 64>,
        cutlass::gemm::GemmShape<64, 64, 64>,
        5,
        cutlass::arch::OpMultiplyAddFastAccum>;
    run_cutlass_gemm<Gemm>(a, b, out, x_scale, w_scale);
  } else {
    using Gemm = W8A8Gemm<
        cutlass::gemm::GemmShape<128, 64, 128>,
        cutlass::gemm::GemmShape<64, 64, 64>,
        3,
        cutlass::arch::OpMultiplyAddFastAccum>;
    run_cutlass_gemm<Gemm>(a, b, out, x_scale, w_scale);
  }
}

void dispatch_m64(
    const torch::Tensor& a,
    const torch::Tensor& b,
    torch::Tensor& out,
    const torch::Tensor& x_scale,
    const torch::Tensor& w_scale) {
  const uint32_t n_pow2 = next_pow2(static_cast<uint32_t>(out.size(1)));
  if (n_pow2 <= 8192) {
    using Gemm = W8A8Gemm<
        cutlass::gemm::GemmShape<64, 64, 128>,
        cutlass::gemm::GemmShape<32, 64, 64>,
        5,
        cutlass::arch::OpMultiplyAdd>;
    run_cutlass_gemm<Gemm>(a, b, out, x_scale, w_scale);
  } else if (n_pow2 <= 16384) {
    using Gemm = W8A8Gemm<
        cutlass::gemm::GemmShape<64, 128, 128>,
        cutlass::gemm::GemmShape<64, 64, 64>,
        3,
        cutlass::arch::OpMultiplyAddFastAccum>;
    run_cutlass_gemm<Gemm>(a, b, out, x_scale, w_scale);
  } else {
    using Gemm = W8A8Gemm<
        cutlass::gemm::GemmShape<64, 64, 128>,
        cutlass::gemm::GemmShape<32, 64, 64>,
        5,
        cutlass::arch::OpMultiplyAdd>;
    run_cutlass_gemm<Gemm>(a, b, out, x_scale, w_scale);
  }
}

void dispatch_m32(
    const torch::Tensor& a,
    const torch::Tensor& b,
    torch::Tensor& out,
    const torch::Tensor& x_scale,
    const torch::Tensor& w_scale) {
  const uint32_t n_pow2 = next_pow2(static_cast<uint32_t>(out.size(1)));
  if (n_pow2 <= 8192) {
    using Gemm = W8A8Gemm<
        cutlass::gemm::GemmShape<32, 64, 128>,
        cutlass::gemm::GemmShape<16, 64, 64>,
        5,
        cutlass::arch::OpMultiplyAddFastAccum>;
    run_cutlass_gemm<Gemm>(a, b, out, x_scale, w_scale);
  } else if (n_pow2 <= 16384) {
    using Gemm = W8A8Gemm<
        cutlass::gemm::GemmShape<32, 128, 128>,
        cutlass::gemm::GemmShape<32, 64, 64>,
        4,
        cutlass::arch::OpMultiplyAddFastAccum>;
    run_cutlass_gemm<Gemm>(a, b, out, x_scale, w_scale);
  } else {
    using Gemm = W8A8Gemm<
        cutlass::gemm::GemmShape<32, 64, 128>,
        cutlass::gemm::GemmShape<16, 64, 64>,
        5,
        cutlass::arch::OpMultiplyAddFastAccum>;
    run_cutlass_gemm<Gemm>(a, b, out, x_scale, w_scale);
  }
}

void dispatch_m16(
    const torch::Tensor& a,
    const torch::Tensor& b,
    torch::Tensor& out,
    const torch::Tensor& x_scale,
    const torch::Tensor& w_scale) {
  const uint32_t n_pow2 = next_pow2(static_cast<uint32_t>(out.size(1)));
  if (n_pow2 <= 8192) {
    using Gemm = W8A8Gemm<
        cutlass::gemm::GemmShape<16, 64, 128>,
        cutlass::gemm::GemmShape<16, 64, 64>,
        5,
        cutlass::arch::OpMultiplyAddFastAccum>;
    run_cutlass_gemm<Gemm>(a, b, out, x_scale, w_scale);
  } else if (n_pow2 <= 24576) {
    using Gemm = W8A8Gemm<
        cutlass::gemm::GemmShape<16, 128, 64>,
        cutlass::gemm::GemmShape<16, 64, 64>,
        5,
        cutlass::arch::OpMultiplyAddFastAccum>;
    run_cutlass_gemm<Gemm>(a, b, out, x_scale, w_scale);
  } else {
    using Gemm = W8A8Gemm<
        cutlass::gemm::GemmShape<32, 64, 128>,
        cutlass::gemm::GemmShape<16, 64, 64>,
        5,
        cutlass::arch::OpMultiplyAddFastAccum>;
    run_cutlass_gemm<Gemm>(a, b, out, x_scale, w_scale);
  }
}

void dispatch_cutlass(
    const torch::Tensor& a,
    const torch::Tensor& b,
    torch::Tensor& out,
    const torch::Tensor& x_scale,
    const torch::Tensor& w_scale) {
  const uint32_t m_pow2 = std::max<uint32_t>(16, next_pow2(static_cast<uint32_t>(a.size(0))));
  if (m_pow2 <= 16) {
    dispatch_m16(a, b, out, x_scale, w_scale);
  } else if (m_pow2 <= 32) {
    dispatch_m32(a, b, out, x_scale, w_scale);
  } else if (m_pow2 <= 64) {
    dispatch_m64(a, b, out, x_scale, w_scale);
  } else if (m_pow2 <= 128) {
    dispatch_m128(a, b, out, x_scale, w_scale);
  } else if (m_pow2 <= 256) {
    dispatch_m256(a, b, out, x_scale, w_scale);
  } else {
    dispatch_default<ElementC>(a, b, out, x_scale, w_scale);
  }
}

__global__ void quantize_activation_kernel(
    const __nv_bfloat16* __restrict__ x,
    __nv_fp8_e4m3* __restrict__ out,
    const float* __restrict__ scale_ptr,
    int numel) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const float scale = fmaxf(scale_ptr[0], 1.0e-12f);
  if (idx < numel) {
    float val = __bfloat162float(x[idx]) / scale;
    val = fminf(fmaxf(val, -448.0f), 448.0f);
    out[idx] = __nv_fp8_e4m3(val);
  }
}

void launch_quantize_activation(const torch::Tensor& x, torch::Tensor& out, const torch::Tensor& scale) {
  const int numel = x.numel();
  const int threads = 256;
  const int blocks = (numel + threads - 1) / threads;
  quantize_activation_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
      reinterpret_cast<__nv_fp8_e4m3*>(out.data_ptr<at::Float8_e4m3fn>()),
      scale.data_ptr<float>(),
      numel);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

__global__ void add_bias_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ bias,
    int m,
    int n) {
  const int row = blockIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < m && col < n) {
    float val = __bfloat162float(out[row * n + col]) + __bfloat162float(bias[col]);
    out[row * n + col] = __float2bfloat16(val);
  }
}

void launch_add_bias(torch::Tensor& out, const torch::Tensor& bias) {
  const int m = out.size(0);
  const int n = out.size(1);
  const int threads = 256;
  dim3 blocks((n + threads - 1) / threads, m);
  add_bias_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(bias.data_ptr<at::BFloat16>()),
      m,
      n);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

torch::Tensor w8a8_cutlass_gemm(
    torch::Tensor x,
    torch::Tensor w,
    torch::Tensor x_scale,
    torch::Tensor w_scale,
    c10::optional<torch::Tensor> bias) {
  TORCH_CHECK(x.is_cuda(), "x must be CUDA tensor");
  TORCH_CHECK(w.is_cuda(), "w must be CUDA tensor");
  TORCH_CHECK(x_scale.is_cuda(), "x_scale must be CUDA tensor");
  TORCH_CHECK(w_scale.is_cuda(), "w_scale must be CUDA tensor");
  TORCH_CHECK(x.dim() == 2, "x must be 2D [M, K]");
  TORCH_CHECK(w.dim() == 2, "w must be 2D [K, N]");
  TORCH_CHECK(x.scalar_type() == torch::kBFloat16, "x must be bfloat16");
  TORCH_CHECK(w.scalar_type() == torch::kFloat8_e4m3fn, "w must be float8_e4m3fn");
  TORCH_CHECK(x_scale.scalar_type() == torch::kFloat32, "x_scale must be float32");
  TORCH_CHECK(w_scale.scalar_type() == torch::kFloat32, "w_scale must be float32");
  TORCH_CHECK(w.size(0) == x.size(1), "w K dimension must match x K dimension");
  TORCH_CHECK(x.stride(1) == 1, "x must be row-major contiguous in K");
  TORCH_CHECK(w.stride(0) == 1, "w must be column-major with stride(0)=1");

  const int64_t m = x.size(0);
  const int64_t n = w.size(1);
  auto out = torch::empty({m, n}, torch::TensorOptions().dtype(torch::kBFloat16).device(x.device()));

  auto x_fp8 = torch::empty_like(x, torch::dtype(torch::kFloat8_e4m3fn));
  x_scale = x_scale.reshape({}).contiguous();
  w_scale = w_scale.reshape({}).contiguous();
  launch_quantize_activation(x, x_fp8, x_scale);
  dispatch_cutlass(x_fp8, w, out, x_scale, w_scale);

  if (bias.has_value() && bias.value().defined()) {
    TORCH_CHECK(bias.value().is_cuda(), "bias must be CUDA tensor");
    TORCH_CHECK(bias.value().scalar_type() == torch::kBFloat16, "bias must be bfloat16");
    TORCH_CHECK(bias.value().numel() == n, "bias size must match output dimension");
    launch_add_bias(out, bias.value());
  }
  return out;
}
