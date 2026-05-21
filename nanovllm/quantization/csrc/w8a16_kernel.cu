#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <torch/extension.h>

namespace {

constexpr int TILE_M = 16;
constexpr int TILE_N = 16;
constexpr int BLOCK_N = 64;
constexpr int TILE_K = 16;
constexpr int WARPS_PER_BLOCK = BLOCK_N / TILE_N;

using namespace nvcuda;

__global__ void w8a16_wmma_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_fp8_e4m3* __restrict__ w,
    const float* __restrict__ scale,
    const __nv_bfloat16* __restrict__ bias,
    __nv_bfloat16* __restrict__ out,
    int M,
    int N,
    int K) {
  __shared__ __half a_s[TILE_M * TILE_K];
  __shared__ __half b_s[TILE_K * BLOCK_N];
  __shared__ float c_s[TILE_M * BLOCK_N];

  const int warp_id = threadIdx.x / warpSize;
  if (warp_id >= WARPS_PER_BLOCK) {
    return;
  }
  const int lane = threadIdx.x % warpSize;
  const int m0 = blockIdx.y * TILE_M;
  const int n0 = blockIdx.x * BLOCK_N;
  const int warp_n0 = warp_id * TILE_N;

  wmma::fragment<wmma::matrix_a, TILE_M, TILE_N, TILE_K, __half, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, TILE_M, TILE_N, TILE_K, __half, wmma::row_major> b_frag;
  wmma::fragment<wmma::accumulator, TILE_M, TILE_N, TILE_K, float> c_frag;
  wmma::fill_fragment(c_frag, 0.0f);

  // Four warps compute one 16x64 output tile and share the same A tile.
  for (int k0 = 0; k0 < K; k0 += TILE_K) {
    for (int idx = threadIdx.x; idx < TILE_M * TILE_K; idx += blockDim.x) {
      const int mi = idx / TILE_K;
      const int ki = idx % TILE_K;
      const int m = m0 + mi;
      const int k = k0 + ki;
      a_s[idx] = (m < M && k < K) ? __float2half(__bfloat162float(x[m * K + k])) : __float2half(0.0f);
    }
    for (int idx = threadIdx.x; idx < TILE_K * BLOCK_N; idx += blockDim.x) {
      const int ki = idx / BLOCK_N;
      const int ni = idx % BLOCK_N;
      const int k = k0 + ki;
      const int n = n0 + ni;
      b_s[idx] = (k < K && n < N) ? __float2half(static_cast<float>(w[k * N + n])) : __float2half(0.0f);
    }
    __syncthreads();
    wmma::load_matrix_sync(a_frag, a_s, TILE_K);
    wmma::load_matrix_sync(b_frag, b_s + warp_n0, BLOCK_N);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    __syncthreads();
  }

  wmma::store_matrix_sync(c_s + warp_n0, c_frag, BLOCK_N, wmma::mem_row_major);
  __syncthreads();
  for (int idx = threadIdx.x; idx < TILE_M * BLOCK_N; idx += blockDim.x) {
    const int mi = idx / BLOCK_N;
    const int ni = idx % BLOCK_N;
    const int m = m0 + mi;
    const int n = n0 + ni;
    if (m < M && n < N) {
      float value = c_s[idx] * scale[n];
      if (bias != nullptr) {
        value += __bfloat162float(bias[n]);
      }
      out[m * N + n] = __float2bfloat16(value);
    }
  }
}

}  // namespace

torch::Tensor w8a16_cuda_forward(
    torch::Tensor x,
    torch::Tensor w,
    torch::Tensor scale,
    c10::optional<torch::Tensor> bias) {
  TORCH_CHECK(x.is_cuda(), "x must be CUDA");
  TORCH_CHECK(w.is_cuda(), "w must be CUDA");
  TORCH_CHECK(scale.is_cuda(), "scale must be CUDA");
  TORCH_CHECK(x.dim() == 2, "x must be 2D");
  TORCH_CHECK(w.dim() == 2, "w must be 2D");
  TORCH_CHECK(x.scalar_type() == torch::kBFloat16, "x must be bfloat16");
  TORCH_CHECK(w.scalar_type() == torch::kFloat8_e4m3fn, "w must be float8_e4m3fn");
  TORCH_CHECK(scale.scalar_type() == torch::kFloat32, "scale must be float32");
  TORCH_CHECK(x.size(1) == w.size(0), "x K and w K must match");
  TORCH_CHECK(scale.numel() == w.size(1), "scale size must match output dimension");
  TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
  TORCH_CHECK(w.is_contiguous(), "w must be contiguous");
  TORCH_CHECK(scale.is_contiguous(), "scale must be contiguous");

  const auto M = static_cast<int>(x.size(0));
  const auto K = static_cast<int>(x.size(1));
  const auto N = static_cast<int>(w.size(1));
  auto out = torch::empty({M, N}, x.options());

  const __nv_bfloat16* bias_ptr = nullptr;
  if (bias.has_value() && bias.value().defined()) {
    TORCH_CHECK(bias.value().is_cuda(), "bias must be CUDA");
    TORCH_CHECK(bias.value().scalar_type() == torch::kBFloat16, "bias must be bfloat16");
    TORCH_CHECK(bias.value().numel() == N, "bias size must match output dimension");
    TORCH_CHECK(bias.value().is_contiguous(), "bias must be contiguous");
    bias_ptr = reinterpret_cast<const __nv_bfloat16*>(bias.value().data_ptr());
  }

  dim3 block(WARPS_PER_BLOCK * 32);
  dim3 grid((N + BLOCK_N - 1) / BLOCK_N, (M + TILE_M - 1) / TILE_M);
  auto stream = at::cuda::getCurrentCUDAStream();
  w8a16_wmma_kernel<<<grid, block, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
      reinterpret_cast<const __nv_fp8_e4m3*>(w.data_ptr()),
      scale.data_ptr<float>(),
      bias_ptr,
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
      M,
      N,
      K);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
