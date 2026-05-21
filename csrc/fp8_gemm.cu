// CUTLASS FP8 GEMM kernel for SM89 (RTX 4090)
// 替代 torch._scaled_mm 的独立 C++ 实现

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/default_gemm_universal.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/numeric_types.h>

// FP8 GEMM configuration for SM89
using ElementA = cutlass::float_e4m3_t;
using ElementB = cutlass::float_e4m3_t;
using ElementC = cutlass::bfloat16_t;
using ElementAccumulator = float;
using ElementCompute = float;

// Kernel template
using GemmKernel = typename cutlass::gemm::kernel::DefaultGemmUniversal<
    ElementA, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kIdentity, 16,
    ElementB, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kIdentity, 16,
    ElementC, cutlass::layout::RowMajor,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm89,
    cutlass::gemm::GemmShape<128, 128, 64>,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<16, 8, 32>,
    cutlass::epilogue::thread::LinearCombination<
        ElementC, 128 / cutlass::sizeof_bits<ElementC>::value,
        ElementAccumulator, ElementCompute>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    3,
    cutlass::arch::OpMultiplyAdd
>::GemmKernel;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

// Forward declaration
torch::Tensor fp8_gemm_cutlass(
    torch::Tensor a,  // FP8 [M, K]
    torch::Tensor b,  // FP8 [K, N]
    torch::Tensor a_scale,  // float scalar
    torch::Tensor b_scale   // float scalar or [N]
);

torch::Tensor fp8_gemm_cutlass(
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor a_scale,
    torch::Tensor b_scale
) {
    // Validate inputs
    TORCH_CHECK(a.is_cuda(), "A must be CUDA tensor");
    TORCH_CHECK(b.is_cuda(), "B must be CUDA tensor");
    TORCH_CHECK(a.scalar_type() == torch::kFloat8_e4m3fn, "A must be FP8");
    TORCH_CHECK(b.scalar_type() == torch::kFloat8_e4m3fn, "B must be FP8");

    int64_t M = a.size(0);
    int64_t K = a.size(1);
    int64_t N = b.size(1);

    // Allocate output
    auto options = torch::TensorOptions()
        .dtype(torch::kBFloat16)
        .device(a.device());
    torch::Tensor c = torch::empty({M, N}, options);

    // Get raw pointers
    auto a_ptr = a.data_ptr<cutlass::float_e4m3_t>();
    auto b_ptr = b.data_ptr<cutlass::float_e4m3_t>();
    auto c_ptr = c.data_ptr<cutlass::bfloat16_t>();

    float alpha = a_scale.item<float>() * b_scale.item<float>();

    // Setup CUTLASS arguments
    typename Gemm::Arguments args(
        {static_cast<int>(M), static_cast<int>(N), static_cast<int>(K)},
        {a_ptr, K},
        {b_ptr, N},
        {c_ptr, N},
        {c_ptr, N},
        {alpha, 0.0f}
    );

    // Run GEMM
    Gemm gemm_op;
    size_t workspace_size = gemm_op.get_workspace_size(args);
    auto workspace = torch::empty({static_cast<int64_t>(workspace_size)},
        torch::TensorOptions().dtype(torch::kUInt8).device(a.device()));

    cutlass::Status status = gemm_op(args, workspace.data_ptr());
    TORCH_CHECK(status == cutlass::Status::kSuccess,
                "CUTLASS FP8 GEMM failed");

    return c;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fp8_gemm_cutlass", &fp8_gemm_cutlass, "CUTLASS FP8 GEMM (SM89)");
}
