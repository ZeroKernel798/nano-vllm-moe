// C++ extension entry point for W8A8 CUTLASS GEMM

#include <torch/extension.h>

// Forward declaration
torch::Tensor w8a8_cutlass_gemm(
    torch::Tensor x,
    torch::Tensor w,
    torch::Tensor x_scale,
    torch::Tensor w_scale,
    c10::optional<torch::Tensor> bias
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("w8a8_cutlass_gemm", &w8a8_cutlass_gemm, "CUTLASS FP8 GEMM for W8A8 (SM89)");
}
