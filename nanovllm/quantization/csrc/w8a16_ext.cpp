#include <torch/extension.h>

torch::Tensor w8a16_cuda_forward(
    torch::Tensor x,
    torch::Tensor w,
    torch::Tensor scale,
    c10::optional<torch::Tensor> bias);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("w8a16_cuda_forward", &w8a16_cuda_forward, "W8A16 CUDA forward");
}
