from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import os
from pathlib import Path
import sys

import torch
from torch.utils.cpp_extension import load_inline


@dataclass(frozen=True)
class W8A8JitRecipe:
    compiled_m: int
    n: int
    k: int
    block_m: int
    block_n: int
    threads: int
    kernel: str

    @property
    def name(self) -> str:
        return (
            "nanovllm_w8a8_jit_"
            f"{self.kernel}_"
            f"m{self.compiled_m}_n{self.n}_k{self.k}_"
            f"bm{self.block_m}_bn{self.block_n}_t{self.threads}"
        )


def _compiled_m(runtime_m: int) -> int:
    bucket = int(os.environ.get("NANOVLLM_W8A8_JIT_M_BUCKET", "16"))
    if bucket <= 0:
        return runtime_m
    return ((runtime_m + bucket - 1) // bucket) * bucket


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return default if value is None or value == "" else int(value)


def _tile_config(runtime_m: int, compiled_m: int) -> tuple[int, int, int]:
    del compiled_m
    # vLLM's CUTLASS FP8 dispatch treats small-M GEMMs as distinct recipes
    # instead of pushing every shape through the default tile.
    if runtime_m <= 1:
        default_block_m = 1
        default_block_n = 8
    elif runtime_m <= 16:
        default_block_m = 4
        default_block_n = 4
    elif runtime_m <= 64:
        default_block_m = 4
        default_block_n = 8
    else:
        default_block_m = 1
        default_block_n = 8
    block_m = _env_int("NANOVLLM_W8A8_JIT_BLOCK_M", default_block_m)
    block_n = _env_int("NANOVLLM_W8A8_JIT_BLOCK_N", default_block_n)
    threads = _env_int("NANOVLLM_W8A8_JIT_THREADS", 128)
    if block_m <= 0 or block_n <= 0 or threads <= 0 or threads > 1024:
        raise ValueError(
            "Invalid W8A8 JIT tile config: "
            f"BLOCK_M={block_m}, BLOCK_N={block_n}, THREADS={threads}"
        )
    if threads % block_m:
        raise ValueError(
            "NANOVLLM_W8A8_JIT_THREADS must be divisible by BLOCK_M, "
            f"got THREADS={threads}, BLOCK_M={block_m}"
        )
    if threads & (threads - 1):
        raise ValueError(f"NANOVLLM_W8A8_JIT_THREADS must be a power of two, got {threads}")
    return block_m, block_n, threads


def _make_recipe(runtime_m: int, n: int, k: int) -> W8A8JitRecipe:
    kernel = os.environ.get("NANOVLLM_W8A8_JIT_KERNEL", "scalar").strip().lower()
    if kernel == "auto_7b":
        if runtime_m == 1:
            kernel = "scalar"
        elif runtime_m <= 32:
            kernel = "ptx_mma_cta_n"
        else:
            raise ValueError(
                "NANOVLLM_W8A8_JIT_KERNEL=auto_7b currently specializes M=1 and 2<=M<=32; "
                f"got runtime M={runtime_m}"
            )
    if kernel not in {"scalar", "ptx_mma", "ptx_mma_cta_n"}:
        raise ValueError(f"Unsupported NANOVLLM_W8A8_JIT_KERNEL={kernel!r}")
    compiled_m = _compiled_m(runtime_m)
    if kernel in {"ptx_mma", "ptx_mma_cta_n"}:
        if k % 32:
            raise ValueError(f"PTX MMA W8A8 kernel requires K divisible by 32, got {k}")
        b_layout = os.environ.get("NANOVLLM_W8A8_JIT_B_LAYOUT", "cutlass").strip().lower()
        if b_layout not in {"cutlass", "ldmatrix_trans"}:
            raise ValueError(f"Unsupported NANOVLLM_W8A8_JIT_B_LAYOUT={b_layout!r}")
        kernel = f"{kernel}_{b_layout}"
        if kernel.startswith("ptx_mma_cta_n"):
            warps_n = _env_int("NANOVLLM_W8A8_JIT_CTA_N_WARPS", 8)
            if warps_n not in {1, 2, 4, 8}:
                raise ValueError(
                    "NANOVLLM_W8A8_JIT_CTA_N_WARPS must be one of 1, 2, 4, 8, "
                    f"got {warps_n}"
                )
            block_m, block_n, threads = 16, warps_n * 8, warps_n * 32
        else:
            block_m, block_n, threads = 16, 8, 32
    else:
        block_m, block_n, threads = _tile_config(runtime_m, compiled_m)
    return W8A8JitRecipe(
        compiled_m=compiled_m,
        n=n,
        k=k,
        block_m=block_m,
        block_n=block_n,
        threads=threads,
        kernel=kernel,
    )


def _ensure_env_path() -> None:
    env_bin = str(Path(sys.executable).resolve().parent)
    path_items = os.environ.get("PATH", "").split(os.pathsep)
    if env_bin not in path_items:
        os.environ["PATH"] = os.pathsep.join([env_bin, *path_items])
    if "TORCH_CUDA_ARCH_LIST" not in os.environ and torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"


def _cuda_source(recipe: W8A8JitRecipe) -> str:
    if recipe.kernel.startswith("ptx_mma"):
        return _cuda_source_ptx_mma(recipe)
    return _cuda_source_scalar(recipe)


def _cuda_source_ptx_mma(recipe: W8A8JitRecipe) -> str:
    # FP8 tensor-core kernel for one warp per 16x8 output tile.
    # CUTLASS cute/atom/mma_traits_sm89.hpp defines the B operand layout as
    # Layout<Shape<Shape<_4,_8>, Shape<_4,_2>>, Stride<Stride<_32,_1>, Stride<_8,_128>>>.
    # For lane t, this maps the two B registers to column t/4 and K offsets
    # 4*(t%4)+[0:4] and +16.  We use that layout directly for B while keeping
    # A loaded through ldmatrix.x4.
    use_cutlass_b = recipe.kernel.endswith("_cutlass")
    return f"""
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <torch/extension.h>

namespace {{

constexpr int COMPILED_M = {recipe.compiled_m};
constexpr int COMPILED_N = {recipe.n};
constexpr int COMPILED_K = {recipe.k};
constexpr int BLOCK_M = 16;
constexpr int BLOCK_N = {recipe.block_n};
constexpr int THREADS = {recipe.threads};
constexpr int WARPS_N = BLOCK_N / 8;

__device__ __forceinline__ uint32_t smem_ptr_to_uint(void const* ptr) {{
  return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}}

__device__ __forceinline__ void ldmatrix_x4(uint32_t r[4], void const* ptr) {{
  uint32_t smem_ptr = smem_ptr_to_uint(ptr);
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {{%0, %1, %2, %3}}, [%4];"
      : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
      : "r"(smem_ptr));
}}

__device__ __forceinline__ void ldmatrix_x2_trans(uint32_t r[2], void const* ptr) {{
  uint32_t smem_ptr = smem_ptr_to_uint(ptr);
  asm volatile(
      "ldmatrix.sync.aligned.trans.m8n8.x2.shared.b16 {{%0, %1}}, [%2];"
      : "=r"(r[0]), "=r"(r[1])
      : "r"(smem_ptr));
}}

__device__ __forceinline__ void ldmatrix_x2(uint32_t r[2], void const* ptr) {{
  uint32_t smem_ptr = smem_ptr_to_uint(ptr);
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {{%0, %1}}, [%2];"
      : "=r"(r[0]), "=r"(r[1])
      : "r"(smem_ptr));
}}

__device__ __forceinline__ uint16_t pack_fp8x2(const __nv_fp8_e4m3 lo, const __nv_fp8_e4m3 hi) {{
  uint16_t out;
  reinterpret_cast<unsigned char*>(&out)[0] = *reinterpret_cast<const unsigned char*>(&lo);
  reinterpret_cast<unsigned char*>(&out)[1] = *reinterpret_cast<const unsigned char*>(&hi);
  return out;
}}

__device__ __forceinline__ uint32_t pack_fp8x4(const __nv_fp8_e4m3* ptr) {{
  return *reinterpret_cast<const uint32_t*>(ptr);
}}

__device__ __forceinline__ void mmaF8_k32(
    float c[4],
    uint32_t const a[4],
    uint32_t const b[2]) {{
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
      "{{%0, %1, %2, %3}}, "
      "{{%4, %5, %6, %7}}, "
      "{{%8, %9}}, "
      "{{%0, %1, %2, %3}};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}}

__global__ __launch_bounds__(THREADS) void w8a8_static_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_fp8_e4m3* __restrict__ w_nk,
    const float* __restrict__ input_scale,
    const float* __restrict__ weight_scale,
    const __nv_bfloat16* __restrict__ bias,
    __nv_bfloat16* __restrict__ out,
    int runtime_m) {{
  const int lane = threadIdx.x & 31;
  const int warp_n = threadIdx.x >> 5;
  const int tid = threadIdx.x;
  const int tile_n = blockIdx.x * BLOCK_N;
  const int tile_m = blockIdx.y * BLOCK_M;
  const float x_scale = fmaxf(input_scale[0], 1.0e-12f);
  const float w_scale = weight_scale[0];

  __shared__ alignas(16) uint16_t a_smem[16][16];

  float c[4] = {{0.f, 0.f, 0.f, 0.f}};

  for (int k0 = 0; k0 < COMPILED_K; k0 += 32) {{
    for (int idx = tid; idx < BLOCK_M * 16; idx += THREADS) {{
      const int row_local = idx / 16;
      const int k_pair = idx % 16;
      const int row_global = tile_m + row_local;
      __nv_fp8_e4m3 q0(0.0f);
      __nv_fp8_e4m3 q1(0.0f);
      if (row_global < runtime_m) {{
        float a0 = __bfloat162float(x[row_global * COMPILED_K + k0 + k_pair * 2 + 0]) / x_scale;
        float a1 = __bfloat162float(x[row_global * COMPILED_K + k0 + k_pair * 2 + 1]) / x_scale;
        q0 = __nv_fp8_e4m3(fminf(fmaxf(a0, -448.0f), 448.0f));
        q1 = __nv_fp8_e4m3(fminf(fmaxf(a1, -448.0f), 448.0f));
      }}
      a_smem[row_local][k_pair] = pack_fp8x2(q0, q1);
    }}
    __syncthreads();

    uint32_t a_frag[4];
    uint32_t b_frag[2];
    ldmatrix_x4(a_frag, &a_smem[lane % 16][(lane / 16) * 8]);
    const int b_col = tile_n + warp_n * 8 + lane / 4;
    const int b_k0 = k0 + (lane % 4) * 4;
    b_frag[0] = 0;
    b_frag[1] = 0;
    if (b_col < COMPILED_N) {{
      b_frag[0] = pack_fp8x4(&w_nk[b_col * COMPILED_K + b_k0]);
      b_frag[1] = pack_fp8x4(&w_nk[b_col * COMPILED_K + b_k0 + 16]);
    }}
    mmaF8_k32(c, a_frag, b_frag);
    __syncthreads();
  }}

  const int out_row0 = tile_m + lane / 4;
  const int out_row1 = out_row0 + 8;
  const int out_col0 = tile_n + warp_n * 8 + (lane % 4) * 2;
  const int out_col1 = out_col0 + 1;

  if (out_row0 < runtime_m && out_col1 < COMPILED_N) {{
    float val;
    val = c[0] * x_scale * w_scale;
    if (bias != nullptr) val += __bfloat162float(bias[out_col0]);
    out[out_row0 * COMPILED_N + out_col0] = __float2bfloat16(val);

    val = c[1] * x_scale * w_scale;
    if (bias != nullptr) val += __bfloat162float(bias[out_col1]);
    out[out_row0 * COMPILED_N + out_col1] = __float2bfloat16(val);
  }}

  if (out_row1 < runtime_m && out_col1 < COMPILED_N) {{
    float val;
    val = c[2] * x_scale * w_scale;
    if (bias != nullptr) val += __bfloat162float(bias[out_col0]);
    out[out_row1 * COMPILED_N + out_col0] = __float2bfloat16(val);

    val = c[3] * x_scale * w_scale;
    if (bias != nullptr) val += __bfloat162float(bias[out_col1]);
    out[out_row1 * COMPILED_N + out_col1] = __float2bfloat16(val);
  }}
}}

}}  // namespace

torch::Tensor w8a8_static_forward(
    torch::Tensor x,
    torch::Tensor w_nk,
    torch::Tensor input_scale,
    torch::Tensor weight_scale,
    c10::optional<torch::Tensor> bias) {{
  TORCH_CHECK(x.is_cuda(), "x must be CUDA");
  TORCH_CHECK(w_nk.is_cuda(), "w_nk must be CUDA");
  TORCH_CHECK(input_scale.is_cuda(), "input_scale must be CUDA");
  TORCH_CHECK(weight_scale.is_cuda(), "weight_scale must be CUDA");
  TORCH_CHECK(x.dim() == 2, "x must be 2D");
  TORCH_CHECK(w_nk.dim() == 2, "w_nk must be 2D");
  TORCH_CHECK(x.scalar_type() == torch::kBFloat16, "x must be bfloat16");
  TORCH_CHECK(w_nk.scalar_type() == torch::kFloat8_e4m3fn, "w_nk must be float8_e4m3fn");
  TORCH_CHECK(x.size(1) == COMPILED_K, "x K does not match compiled K");
  TORCH_CHECK(w_nk.size(0) == COMPILED_N, "w_nk N does not match compiled N");
  TORCH_CHECK(w_nk.size(1) == COMPILED_K, "w_nk K does not match compiled K");
  TORCH_CHECK(x.size(0) <= COMPILED_M, "runtime M exceeds compiled M");
  x = x.contiguous();
  w_nk = w_nk.contiguous();
  input_scale = input_scale.reshape({{}}).to(torch::kFloat32).contiguous();
  weight_scale = weight_scale.reshape({{}}).to(torch::kFloat32).contiguous();
  const auto runtime_m = static_cast<int>(x.size(0));
  auto out = torch::empty({{runtime_m, COMPILED_N}}, x.options());
  const __nv_bfloat16* bias_ptr = nullptr;
  if (bias.has_value() && bias.value().defined()) {{
    TORCH_CHECK(bias.value().is_cuda(), "bias must be CUDA");
    TORCH_CHECK(bias.value().scalar_type() == torch::kBFloat16, "bias must be bfloat16");
    TORCH_CHECK(bias.value().numel() == COMPILED_N, "bias size must match output dimension");
    bias_ptr = reinterpret_cast<const __nv_bfloat16*>(bias.value().contiguous().data_ptr());
  }}
  dim3 block(THREADS);
  dim3 grid((COMPILED_N + BLOCK_N - 1) / BLOCK_N, (runtime_m + BLOCK_M - 1) / BLOCK_M);
  auto stream = at::cuda::getCurrentCUDAStream();
  w8a8_static_kernel<<<grid, block, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
      reinterpret_cast<const __nv_fp8_e4m3*>(w_nk.data_ptr()),
      input_scale.data_ptr<float>(),
      weight_scale.data_ptr<float>(),
      bias_ptr,
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
      runtime_m);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}}
"""


def _cuda_source_scalar(recipe: W8A8JitRecipe) -> str:
    return f"""
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <torch/extension.h>

namespace {{

constexpr int COMPILED_M = {recipe.compiled_m};
constexpr int COMPILED_N = {recipe.n};
constexpr int COMPILED_K = {recipe.k};
constexpr int BLOCK_M = {recipe.block_m};
constexpr int BLOCK_N = {recipe.block_n};
constexpr int THREADS = {recipe.threads};
constexpr int THREADS_PER_ROW = THREADS / BLOCK_M;

__global__ __launch_bounds__(THREADS) void w8a8_static_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_fp8_e4m3* __restrict__ w_nk,
    const float* __restrict__ input_scale,
    const float* __restrict__ weight_scale,
    const __nv_bfloat16* __restrict__ bias,
    __nv_bfloat16* __restrict__ out,
    int runtime_m) {{
  const int row_base = blockIdx.y * BLOCK_M;
  const int col_base = blockIdx.x * BLOCK_N;

  const float x_scale = fmaxf(input_scale[0], 1.0e-12f);
  const float w_scale = weight_scale[0];
  const int local_row = threadIdx.x / THREADS_PER_ROW;
  const int lane = threadIdx.x - local_row * THREADS_PER_ROW;
  const int row = row_base + local_row;
  float acc[BLOCK_N];
  #pragma unroll
  for (int j = 0; j < BLOCK_N; ++j) {{
    acc[j] = 0.0f;
  }}

  __shared__ float scratch[BLOCK_M][BLOCK_N][THREADS_PER_ROW];
  if (row < runtime_m) {{
    for (int kk = lane; kk < COMPILED_K; kk += THREADS_PER_ROW) {{
      float a = __bfloat162float(x[row * COMPILED_K + kk]) / x_scale;
      a = fminf(fmaxf(a, -448.0f), 448.0f);
      #pragma unroll
      for (int j = 0; j < BLOCK_N; ++j) {{
        const int col = col_base + j;
        if (col < COMPILED_N) {{
          float b = static_cast<float>(w_nk[col * COMPILED_K + kk]);
          acc[j] += a * b;
        }}
      }}
    }}
  }}

  #pragma unroll
  for (int j = 0; j < BLOCK_N; ++j) {{
    scratch[local_row][j][lane] = acc[j];
  }}
  __syncthreads();

  for (int stride = THREADS_PER_ROW / 2; stride > 0; stride >>= 1) {{
    if (lane < stride) {{
      #pragma unroll
      for (int j = 0; j < BLOCK_N; ++j) {{
        scratch[local_row][j][lane] += scratch[local_row][j][lane + stride];
      }}
    }}
    __syncthreads();
  }}

  if (lane == 0 && row < runtime_m) {{
    #pragma unroll
    for (int j = 0; j < BLOCK_N; ++j) {{
      const int col = col_base + j;
      if (col < COMPILED_N) {{
        float value = scratch[local_row][j][0] * x_scale * w_scale;
        if (bias != nullptr) {{
          value += __bfloat162float(bias[col]);
        }}
        out[row * COMPILED_N + col] = __float2bfloat16(value);
      }}
    }}
  }}
}}

}}  // namespace

torch::Tensor w8a8_static_forward(
    torch::Tensor x,
    torch::Tensor w_nk,
    torch::Tensor input_scale,
    torch::Tensor weight_scale,
    c10::optional<torch::Tensor> bias) {{
  TORCH_CHECK(x.is_cuda(), "x must be CUDA");
  TORCH_CHECK(w_nk.is_cuda(), "w_nk must be CUDA");
  TORCH_CHECK(input_scale.is_cuda(), "input_scale must be CUDA");
  TORCH_CHECK(weight_scale.is_cuda(), "weight_scale must be CUDA");
  TORCH_CHECK(x.dim() == 2, "x must be 2D");
  TORCH_CHECK(w_nk.dim() == 2, "w_nk must be 2D");
  TORCH_CHECK(x.scalar_type() == torch::kBFloat16, "x must be bfloat16");
  TORCH_CHECK(w_nk.scalar_type() == torch::kFloat8_e4m3fn, "w_nk must be float8_e4m3fn");
  TORCH_CHECK(input_scale.scalar_type() == torch::kFloat32, "input_scale must be float32");
  TORCH_CHECK(weight_scale.scalar_type() == torch::kFloat32, "weight_scale must be float32");
  TORCH_CHECK(x.size(1) == COMPILED_K, "x K does not match compiled K");
  TORCH_CHECK(w_nk.size(0) == COMPILED_N, "w_nk N does not match compiled N");
  TORCH_CHECK(w_nk.size(1) == COMPILED_K, "w_nk K does not match compiled K");
  TORCH_CHECK(x.size(0) <= COMPILED_M, "runtime M exceeds compiled M");
  TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
  TORCH_CHECK(w_nk.is_contiguous(), "w_nk must be contiguous");
  TORCH_CHECK(input_scale.is_contiguous(), "input_scale must be contiguous");
  TORCH_CHECK(weight_scale.is_contiguous(), "weight_scale must be contiguous");

  const auto runtime_m = static_cast<int>(x.size(0));
  auto out = torch::empty({{runtime_m, COMPILED_N}}, x.options());
  const __nv_bfloat16* bias_ptr = nullptr;
  if (bias.has_value() && bias.value().defined()) {{
    TORCH_CHECK(bias.value().is_cuda(), "bias must be CUDA");
    TORCH_CHECK(bias.value().scalar_type() == torch::kBFloat16, "bias must be bfloat16");
    TORCH_CHECK(bias.value().numel() == COMPILED_N, "bias size must match output dimension");
    TORCH_CHECK(bias.value().is_contiguous(), "bias must be contiguous");
    bias_ptr = reinterpret_cast<const __nv_bfloat16*>(bias.value().data_ptr());
  }}

  dim3 block(THREADS);
  dim3 grid((COMPILED_N + BLOCK_N - 1) / BLOCK_N, (runtime_m + BLOCK_M - 1) / BLOCK_M);
  auto stream = at::cuda::getCurrentCUDAStream();
  w8a8_static_kernel<<<grid, block, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
      reinterpret_cast<const __nv_fp8_e4m3*>(w_nk.data_ptr()),
      input_scale.data_ptr<float>(),
      weight_scale.data_ptr<float>(),
      bias_ptr,
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
      runtime_m);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}}
"""


_CPP_SOURCE = """
#include <torch/extension.h>

torch::Tensor w8a8_static_forward(
    torch::Tensor x,
    torch::Tensor w_nk,
    torch::Tensor input_scale,
    torch::Tensor weight_scale,
    c10::optional<torch::Tensor> bias);
"""


@lru_cache(maxsize=128)
def _load_w8a8_extension(recipe: W8A8JitRecipe):
    _ensure_env_path()
    return load_inline(
        name=recipe.name,
        cpp_sources=[_CPP_SOURCE],
        cuda_sources=[_cuda_source(recipe)],
        functions=["w8a8_static_forward"],
        extra_cflags=["-O3"],
        extra_cuda_cflags=[
            "-O3",
            "--use_fast_math",
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        ],
        with_cuda=True,
        verbose=False,
    )


def launch_w8a8_cuda_ptx_jit(
    x_bf16: torch.Tensor,
    w_fp8_nk: torch.Tensor,
    input_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    if not x_bf16.is_cuda or not w_fp8_nk.is_cuda:
        raise ValueError("W8A8 CUDA/PTX JIT backend requires CUDA tensors")
    if x_bf16.dtype != torch.bfloat16:
        x_bf16 = x_bf16.to(torch.bfloat16)
    x_bf16 = x_bf16.contiguous()
    w_fp8_nk = w_fp8_nk.contiguous()
    input_scale = input_scale.reshape(()).to(torch.float32).contiguous()
    weight_scale = weight_scale.reshape(()).to(torch.float32).contiguous()
    if bias is not None:
        bias = bias.contiguous()
    runtime_m, k = x_bf16.shape
    n = w_fp8_nk.shape[0]
    recipe = _make_recipe(runtime_m, n, k)
    ext = _load_w8a8_extension(recipe)
    return ext.w8a8_static_forward(x_bf16, w_fp8_nk, input_scale, weight_scale, bias)
