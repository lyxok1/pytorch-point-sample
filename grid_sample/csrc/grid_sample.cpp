#include <torch/torch.h>
#include "grid_sample_kernel.cuh"

std::vector<std::string> interp_types = {"nn", "bilinear"};

int resample_cuda_forward(
    at::Tensor& input,
    at::Tensor& points, 
    at::Tensor& output,
    std::string type) {
      resample_kernel_forward(
        input, 
        points, 
        output,
        type
      );
    return 1;
}

int resample_cuda_backward(
    at::Tensor& input, 
    at::Tensor& points,
    at::Tensor& gradOutput,
    at::Tensor& gradInput, 
    at::Tensor& gradPoints,
    std::string type) {
        resample_kernel_backward(
          input, 
          points, 
          gradOutput, 
          gradInput, 
          gradPoints,
          type
        );
    return 1;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &resample_cuda_forward, "Resample forward (CUDA)");
  m.def("backward", &resample_cuda_backward, "Resample backward (CUDA)");
  m.attr("InterpType") = interp_types;
}
