/*

This code is borrowed from the flownet2.0 warping function:
https://github.com/NVIDIA/flownet2-pytorch/blob/master/networks/resample2d_package

*/

#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <stdio.h>

#define CUDA_NUM_THREADS 512 
#define THREADS_PER_BLOCK 64

#define CUDA_GET_BLOCKS(n) ((n + (CUDA_NUM_THREADS) - 1) / (CUDA_NUM_THREADS))

#define TENSOR_INDEX(TENSOR, xx, yy, zz, ww) ((TENSOR)[((xx) * (width * height * channel)) + ((yy) * (width * height)) + ((zz) * (width)) + (ww)])
#define POINTS_INDEX(POINTS, xx, yy, zz) ((POINTS)[((xx) * (npoints * 2)) + ((yy) * 2) + (zz)])
#define OUTPUT_INDEX(OUTPUT, xx, yy, zz) ((OUTPUT)[((xx) * (channel * npoints)) + ((yy) * (npoints)) + (zz)])

#define CUDA_KERNEL_LOOP(index, n) for(int index=blockIdx.x * blockDim.x + threadIdx.x; index < n; index += blockDim.x * gridDim.x)

// resample kernel

template <typename scalar_t>
__global__ void kernel_resample2d_update_output_bilinear(const int n, 
        const scalar_t* __restrict__ input, const scalar_t* __restrict__ points,
        scalar_t* __restrict__ output, const long batchsize, const long channel, 
        const long height, const long width, const long npoints) {
    CUDA_KERNEL_LOOP(index, n)
    {

        long dim_p = npoints * channel;
        long dim_c = npoints;

        long b = ( index / dim_p ) % batchsize;
        long c = ( index / dim_c ) % channel;
        long p = ( index         ) % npoints;

        scalar_t px = POINTS_INDEX(points, b, p, 0);
        scalar_t py = POINTS_INDEX(points, b, p, 1);

        long xL = max(min( long (floor(px)),    width-1), 0L);
        long xR = max(min( long (floor(px)+1),  width-1), 0L);
        long yT = max(min( long (floor(py)),    height-1), 0L);
        long yB = max(min( long (floor(py)+1),  height-1), 0L);

        scalar_t alpha = px >= 0 && px <= (width-1) ?  (1 - (px - floor(px))) : 0.0; // alpha
        scalar_t alpha_bar = px >= 0 && px <= (width-1) ? (1 - alpha) : 0.0;
        scalar_t beta = py >= 0 && py <= (height-1) ? (1 - (py - floor(py))) : 0.0; // beta
        scalar_t beta_bar = py >= 0 && py <= (height-1) ? (1 - beta) : 0.0;

        OUTPUT_INDEX(output, b, c, p)  = TENSOR_INDEX(input, b, c, yT, xL) * (alpha) * (beta);
        OUTPUT_INDEX(output, b, c, p) += TENSOR_INDEX(input, b, c, yT, xR) * (alpha_bar) * (beta);
        OUTPUT_INDEX(output, b, c, p) += TENSOR_INDEX(input, b, c, yB, xL) * (alpha) * (beta_bar);
        OUTPUT_INDEX(output, b, c, p) += TENSOR_INDEX(input, b, c, yB, xR) * (alpha_bar) * (beta_bar);
    }
}

template <typename scalar_t>
__global__ void kernel_resample2d_update_output_nearest(const int n, 
        const scalar_t* __restrict__ input, const scalar_t* __restrict__ points,
        scalar_t* __restrict__ output, const long batchsize, const long channel, 
        const long height, const long width, const long npoints) {
    CUDA_KERNEL_LOOP(index, n)
    {

        long dim_p = npoints * channel;
        long dim_c = npoints;

        long b = ( index / dim_p ) % batchsize;
        long c = ( index / dim_c ) % channel;
        long p = ( index         ) % npoints;

        scalar_t px = POINTS_INDEX(points, b, p, 0);
        scalar_t py = POINTS_INDEX(points, b, p, 1);

        if(px >= 0 && px <= (width - 1) && py >= 0 && py <= (height - 1))
        {
            long xT = (px - floor(px)) < 0.5 ? long(floor(px)) : long(floor(px) + 1);
            long yT = (py - floor(py)) < 0.5 ? long(floor(py)) : long(floor(py) + 1);
            xT = max(0L, min(xT, width-1));
            yT = max(0L, min(yT, height-1));

            OUTPUT_INDEX(output, b, c, p) = TENSOR_INDEX(input, b, c, yT, xT);
        }
        else
            OUTPUT_INDEX(output, b, c, p) = 0;
    }

}

//backward kernel

template <typename scalar_t>
__global__ void kernel_resample2d_backward_input_bilinear(
    const int n, const scalar_t* __restrict__ input, const scalar_t* __restrict__ points, 
    const scalar_t* __restrict__ gradOutput, scalar_t* __restrict__ gradInput,
    const long batchsize, const long channel, const long height, const long width, const long npoints) {

    CUDA_KERNEL_LOOP(index, n)
    {

        long dim_p = npoints * channel;
        long dim_c = npoints;

        long b = ( index / dim_p ) % batchsize;
        long c = ( index / dim_c ) % channel;
        long p = ( index         ) % npoints;

        scalar_t px = POINTS_INDEX(points, b, p, 0);
        scalar_t py = POINTS_INDEX(points, b, p, 1);

        long xL = max(min( long (floor(px)),    width-1), 0L);
        long xR = max(min( long (floor(px)+1),  width-1), 0L);
        long yT = max(min( long (floor(py)),    height-1), 0L);
        long yB = max(min( long (floor(py)+1),  height-1), 0L);

        scalar_t alpha = px >= 0 && px <= (width-1) ?  (1 - (px - floor(px))) : 0.0; // alpha
        scalar_t alpha_bar = px >= 0 && px <= (width-1) ? (1 - alpha) : 0.0;
        scalar_t beta = py >= 0 && py <= (height-1) ? (1 - (py - floor(py))) : 0.0; // beta
        scalar_t beta_bar = py >= 0 && py <= (height-1) ? (1 - beta) : 0.0;

        atomicAdd(&TENSOR_INDEX(gradInput, b, c, yT, xL), OUTPUT_INDEX(gradOutput, b, c, p) * (alpha) * (beta));
        atomicAdd(&TENSOR_INDEX(gradInput, b, c, yT, xR), OUTPUT_INDEX(gradOutput, b, c, p) * (alpha_bar) * (beta));
        atomicAdd(&TENSOR_INDEX(gradInput, b, c, yB, xL), OUTPUT_INDEX(gradOutput, b, c, p) * (alpha) * (beta_bar));
        atomicAdd(&TENSOR_INDEX(gradInput, b, c, yB, xR), OUTPUT_INDEX(gradOutput, b, c, p) * (alpha_bar) * (beta_bar));
    }

}

template <typename scalar_t>
__global__ void kernel_resample2d_backward_input_nearest(
    const int n, const scalar_t* __restrict__ input, const scalar_t* __restrict__ points, 
    const scalar_t* __restrict__ gradOutput, scalar_t* __restrict__ gradInput,
    const long batchsize, const long channel, const long height, const long width, const long npoints) {

    CUDA_KERNEL_LOOP(index, n)
    {

        long dim_p = npoints * channel;
        long dim_c = npoints;

        long b = ( index / dim_p ) % batchsize;
        long c = ( index / dim_c ) % channel;
        long p = ( index         ) % npoints;

        scalar_t px = POINTS_INDEX(points, b, p, 0);
        scalar_t py = POINTS_INDEX(points, b, p, 1);

        if(px >= 0 && px <= (width - 1) && py >= 0 && py <= (height - 1))
        {
            long xT = (px - floor(px)) < 0.5 ? long(floor(px)) : long(floor(px) + 1);
            long yT = (py - floor(py)) < 0.5 ? long(floor(py)) : long(floor(py) + 1);
            xT = max(0L, min(xT, width-1));
            yT = max(0L, min(yT, height-1));
            atomicAdd(&TENSOR_INDEX(gradInput, b, c, yT, xT), OUTPUT_INDEX(gradOutput, b, c, p));
        }
    }

}

template <typename scalar_t>
__global__ void kernel_resample2d_backward_points_bilinear(
    const int n, const scalar_t* __restrict__ input, const scalar_t* __restrict__ points, 
    const scalar_t* __restrict__ gradOutput, scalar_t* __restrict__ gradPoints,
    const long batchsize, const long channel, const long height, const long width, const long npoints) {

    CUDA_KERNEL_LOOP(index, n)
    {
        scalar_t grad_x_val = 0.0f;
        scalar_t grad_y_val = 0.0f;

        long dim_c = npoints;

        long b = ( index / dim_c ) % batchsize;
        long p = ( index         ) % dim_c;

        scalar_t px = POINTS_INDEX(points, b, p, 0);
        scalar_t py = POINTS_INDEX(points, b, p, 1);

        long xL = max(min( long (floor(px)),    width-1), 0L);
        long xR = max(min( long (floor(px)+1),  width-1), 0L);
        long yT = max(min( long (floor(py)),    height-1), 0L);
        long yB = max(min( long (floor(py)+1),  height-1), 0L);

        scalar_t alpha = px >= 0 && px <= (width-1) ?  (1 - (px - floor(px))) : 0.0; // alpha
        scalar_t alpha_bar = px >= 0 && px <= (width-1) ? (1 - alpha) : 0.0;
        scalar_t beta = py >= 0 && py <= (height-1) ? (1 - (py - floor(py))) : 0.0; // beta
        scalar_t beta_bar = py >= 0 && py <= (height-1) ? (1 - beta) : 0.0;

        for(long c=0; c < channel; c++)
        {
            if(px >= 0 && px <= (width - 1))
            {
                grad_x_val -= static_cast<scalar_t>(TENSOR_INDEX(gradOutput, b, c, yT, xL) * (beta));
                grad_x_val += static_cast<scalar_t>(TENSOR_INDEX(gradOutput, b, c, yT, xR) * (beta_bar));
                grad_x_val -= static_cast<scalar_t>(TENSOR_INDEX(gradOutput, b, c, yB, xL) * (beta));
                grad_x_val += static_cast<scalar_t>(TENSOR_INDEX(gradOutput, b, c, yB, xR) * (beta_bar));
            }

            if(py >= 0 && py <= (height - 1))
            {
                grad_y_val -= static_cast<scalar_t>(TENSOR_INDEX(gradOutput, b, c, yT, xL) * (alpha));
                grad_y_val += static_cast<scalar_t>(TENSOR_INDEX(gradOutput, b, c, yT, xR) * (alpha_bar));
                grad_y_val -= static_cast<scalar_t>(TENSOR_INDEX(gradOutput, b, c, yB, xL) * (alpha));
                grad_y_val += static_cast<scalar_t>(TENSOR_INDEX(gradOutput, b, c, yB, xR) * (alpha_bar));
            }
        }
        
        POINTS_INDEX(gradPoints, b, p, 0) = grad_x_val;
        POINTS_INDEX(gradPoints, b, p, 1) = grad_y_val;
    }

}

void resample_kernel_forward(
    at::Tensor& input, 
    at::Tensor& points,
    at::Tensor& output,
    std::string type) {

    long n = output.numel();
    long batchsize = input.size(0);
    long channel = input.size(1);
    long height = input.size(2);
    long width = input.size(3);

    long npoints = points.size(1);

    // TODO: when atomicAdd gets resolved, change to AT_DISPATCH_FLOATING_TYPES_AND_HALF
    if(type == "bilinear")
    {
       {
            kernel_resample2d_update_output_bilinear<float><<< (n + CUDA_NUM_THREADS - 1)/CUDA_NUM_THREADS, CUDA_NUM_THREADS, 0>>>(
                n,
                input.data<float>(),
                points.data<float>(),
                output.data<float>(),
                batchsize,
                channel,
                height,
                width,
                npoints
                );

       }
   }
   else if(type == "nn")
   {
         {
            kernel_resample2d_update_output_nearest<float><<< (n + CUDA_NUM_THREADS - 1)/CUDA_NUM_THREADS, CUDA_NUM_THREADS, 0>>>(
                n,
                input.data<float>(),
                points.data<float>(),
                output.data<float>(),
                batchsize,
                channel,
                height,
                width,
                npoints
                );

       }
   }
   return;
}

void resample_kernel_backward(
    at::Tensor& input,
    at::Tensor& points,
    at::Tensor& gradOutput,
    at::Tensor& gradInput,
    at::Tensor& gradPoints,
    std::string type) {

    long batchsize = input.size(0);
    long channel = input.size(1);
    long height = input.size(2);
    long width = input.size(3);

    long npoints = points.size(1);
    
    if(type == "bilinear")
    {
        long n = gradOutput.numel();

        {

            kernel_resample2d_backward_input_bilinear<float><<< (n + CUDA_NUM_THREADS - 1)/CUDA_NUM_THREADS, CUDA_NUM_THREADS, 0>>>(
                n, 
                input.data<float>(), 
                points.data<float>(),
                gradOutput.data<float>(),
                gradInput.data<float>(),
                batchsize,
                channel,
                height,
                width,
                npoints
            );

        }

        n = batchsize * npoints;

        {
            kernel_resample2d_backward_points_bilinear<float><<< (n + CUDA_NUM_THREADS - 1)/CUDA_NUM_THREADS, CUDA_NUM_THREADS, 0>>>(
                n, 
                input.data<float>(), 
                points.data<float>(),
                gradOutput.data<float>(),
                gradPoints.data<float>(),
                batchsize,
                channel,
                height,
                width,
                npoints
            );

        }

    }
    else if(type == "nn")
    {
        long n = gradOutput.numel();

        {

            kernel_resample2d_backward_input_nearest<float><<< (n + CUDA_NUM_THREADS - 1)/CUDA_NUM_THREADS, CUDA_NUM_THREADS, 0>>>(
                n, 
                input.data<float>(), 
                points.data<float>(),
                gradOutput.data<float>(),
                gradInput.data<float>(),
                batchsize,
                channel,
                height,
                width,
                npoints
            );

        }
    }

    return;
}
