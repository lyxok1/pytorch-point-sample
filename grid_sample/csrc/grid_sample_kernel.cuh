#pragma once

#include <ATen/ATen.h>
#include <string>

void resample_kernel_forward(
    at::Tensor& input,
    at::Tensor& points,
    at::Tensor& output,
    std::string type);

void resample_kernel_backward(
    at::Tensor& input,
    at::Tensor& points,
    at::Tensor& gradOutput,
    at::Tensor& gradInput, 
    at::Tensor& gradPoints,
    std::string type);