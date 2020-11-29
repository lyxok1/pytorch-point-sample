from torch.nn.modules.module import Module
from torch.autograd import Function

import grid_sample
import torch

from grid_sample import _C

class SampleFunction(Function):

    '''
    Sampling Wrapper
    '''

    @staticmethod
    def forward(ctx, inputs, points, interp_type):

        # sanity check
        assert inputs.is_contiguous()
        assert points.is_contiguous()
        assert inputs.is_cuda and points.is_cuda
        assert inputs.size(0) == points.size(0)
        assert points.ndimension() == 3 and points.size(2) == 2
        assert interp_type in _C.InterpType

        ctx.save_for_backward(inputs, points)
        ctx.constant = interp_type

        b, c, h, w = inputs.size()
        p = points.size(1)
        output = inputs.new(b, c, p).zero_()

        _C.forward(inputs, points, output, interp_type)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.contiguous()
        assert grad_output.is_contiguous()

        inputs, points = ctx.saved_tensors
        interp_type = ctx.constant

        grad_inputs = inputs.new(inputs.size()).zero_()
        grad_points = points.new(points.size()).zero_()

        _C.backward(inputs, points, grad_output,
                            grad_inputs, grad_points, interp_type)

        if interp_type == 'nn':
            grad_points = None

        return grad_inputs, grad_points, None

# utility functions

def sample(inputs, points, interp_type ='bilinear'):

    """
    General point sampling interface
    inputs:
        torch.Tensor, input tensor of size [b x c x h x w]
    points:
        torch.Tensor, points index of size [b x p x 2], with coordinate format [x, y]
    interp_type:
        str, string to specify the interpolation type, 'bilinear' or 'nn'
    """

    inputs_c = inputs.contiguous()
    points_c = points.contiguous()
    return SampleFunction.apply(inputs_c, points_c, interp_type)

def make_grid(height, width, device):

    grid_y = torch.linspace(0, height-1, height)
    grid_x = torch.linspace(0, width-1, width)

    grid = torch.stack([
            grid_x.unsqueeze(0).expand(height, width),
            grid_y.unsqueeze(1).expand(height, width),
            ], dim=0)

    grid = grid.to(device)

    return grid

def make_grid_as(tensor):

    """
    tensor: 
        torch.Tensor, with size [b x c x h x w]
    """

    assert tensor.ndimension() == 4

    height, width = tensor.size(2), tensor.size(3)
    grid = make_grid(height, width, tensor.device)
    grid = grid.unsqueeze(0).expand(tensor.size(0), 2, height, width)

    return grid

def make_grid_from_box(boxes, output_size, sample_ratio, align=True):

    """
    boxes: 
        torch.Tensor, with size [k x 4] of format [xmin, ymin, xmax, ymax]
    output_size: 
        tuple[int], output height and width of pooled box
    sample_ratio: 
        int, number of points in each bin on x/y dimension
    align: 
        bool, whether set the grid coordinates with a -0.5 offset to align geometry points and index
    """
    nbox = boxes.shape[0]

    box_w = boxes[:, 2] - boxes[:, 0]
    box_h = boxes[:, 3] - boxes[:, 1]

    bin_x = box_w / output_size[1]
    bin_y = box_h / output_size[0]

    sub_x = bin_x / (sample_ratio + 1)
    sub_y = bin_y / (sample_ratio + 1)

    tx = torch.stack(
            [boxes[:, 0] + i * bin_x for i in range(output_size[1])],
            dim=1
            )
    ty = torch.stack(
            [boxes[:, 1] + i * bin_y for i in range(output_size[0])],
            dim=1
            )

    tx = tx.unsqueeze(1).expand(-1, *output_size)
    ty = ty.unsqueeze(2).expand(-1, *output_size)

    bin_p = torch.stack([tx, ty], dim=-1)
    bin_p = bin_p.view(nbox, -1, 1, 2)
    
    dx = torch.stack([(i + 1) * sub_x for i in range(sample_ratio)], dim=1)
    dy = torch.stack([(i + 1) * sub_y for i in range(sample_ratio)], dim=1)
    dx = tx.unsqueeze(1).expand(-1, sample_ratio, sample_ratio)
    dy = ty.unsqueeze(2).expand(-1, sample_ratio, sample_ratio)

    sp = torch.stack([dx, dy], dim=-1)
    sp = sp.view(nbox, 1, -1, 2)

    grid = bin_p + sp

    if align:
        grid -= 0.5

    grid = grid.view(nbox, *output_size, sample_ratio, sample_ratio, 2)

    return grid

if __name__ == '__main__':

    m = sample

    # test warping function
    input1 = torch.zeros(1, 1, 5, 5).cuda()
    input1[0, 0, 1:3, 1:3] = 1.0
    input1[0, 0, 3, 3] = 2.0
    input1.requires_grad = True

    print('input map: ')
    print(input1)

    points = torch.Tensor([
        [
            [0.4, 1.7],
            [2.3, 3.5],
            [1.8, 4.7]
        ]
    ]).cuda()

    print(points)
    points.requires_grad = True

    out = m(input1, points)

    print('output map')
    print(out)

    loss = torch.mean(out)
    loss.backward()

    print(out.grad)
    print(input1.grad)
    print(points.grad)

