import torch
import grid_sample.utils as utils

class ROIAlign(object):

    """
    Sampling from bins of given ROIs
    The interface is consistent with Detectron2:
    https://github.com/facebookresearch/detectron2
    """
    def __init__(self, output_size, spatial_scale, sampling_ratio, aligned=True):

        self.output_size = output_size
        self.spatial_scale = spatial_scale
        self.sampling_ratio = sampling_ratio
        self.aligned = aligned

    def __call__(self, x, boxes):

        """
        x: torch.Tensor of size [b x c x h x w]
        boxes: torch.Tensor of size [n x 5] with  format [batch_idx, xmin, ymin, xmax, ymax]
        """

        b, c, h, w = x.shape
        boxes[:, 1:] /= self.spatial_scale
        output = []

        for i in range(b):
            box_idx = boxes[:, 0] == i
            if torch.sum(box_idx) == 0:
                continue

            box = boxes[box_idx, 1:]
            nbox = box.shape[0]
            grid = utils.make_grid_from_box(box, self.output_size, sampling_ratio, self.aligned)
            points = grid.view(1, -1, 2).permute(0, 2, 1).contiguous()

            box_out = utils.sample(x[i:i+1], points, 'bilinear')
            box_out = box_out.contiguous().view(c, nbox, *self.output_size, -1)
            box_out, _ = torch.max(box_out, dim=-1)

            output.append(box_out.permute(1, 0, 2).contiguous())
        
        output = torch.cat(output, dim=0)

        return output

class Resize(object):

    """
    Resize the tensor to a given height x width
    """

    def __init__(self, size, interp_type = 'bilinear'):
        if isinstance(size, int):
            self.height, self.width = size, size
        elif isinstance(size, (tuple, list)):
            assert len(size) >= 2
            self.height, self.width = size[0], size[1]
        else:
            raise TypeError('unsupported size type')

        self.interp_type = interp_type

    def __call__(self, x):

        """
        x: torch.Tensor of size [b x c x h x w]
        """
        b, c, h, w = x.shape
        hr = h / self.height
        wr = w / self.width
        grid = utils.make_grid(self.height, self.width, x.device)
        grid[0] *= wr
        grid[1] *= hr

        points = grid.permute(1, 2, 0).contiguous().view(1, -1, 2)
        points = points.expand(b, -1, 2)

        output = utils.sample(x, points, self.interp_type)

        output = output.contiguous().view(b, c, self.height, self.width)

        return output

class Sample(object):

    """
    Sampling the tensor from given points
    """
    def __init__(self, interp_type = 'bilinear'):
        self.interp_type = interp_type

    def __call__(self, x, points):

        """
        x: torch.Tensor of size [b x c x h x w]
        points: torch.Tensor of size [b x p x 2]
        """
        b, c, h, w = x.shape
        output = utils.sample(x, points, self.interp_type)

        return output.contiguous()


class WarpFlow(object):

    """
    Warp the tensor from given flow-field
    """

    def __call__(self, x, flow):

        """
        x: torch.Tensor of size [b x c x h x w]
        flow: torch.Tensor of size [b x 2 x h x w]
        """
        b, c, h, w = x.shape
        grid = utils.make_grid_as(x)

        src = grid + flow
        points = src.permute(0, 2, 3, 1).contiguous().view(b, -1, 2)

        output = utils.sample(x, points, 'bilinear')
        output = output.contiguous().view(b, c, h, w)

        return output

class Affine(object):

    """
    Apply affine transformation from given affine matrix
    """

    def __call__(self, x, affine):

        """
        x: torch.Tensor of size [b x c x h x w]
        affine: torch.Tensor of size [b x 2 x 3]
        """
        b, c, h, w = x.shape
        grid = utils.make_grid_as(x)
        grid = grid.view(b, 2, -1)

        const = grid.new_ones(b, 1, h * w)
        grid = torch.cat([grid, const], dim=1)

        src = torch.matmul(affine, grid)
        points = src.permute(0, 2, 1).contiguous()

        output = utils.sample(x, points, 'bilinear')
        output = output.contiguous().view(b, c, h, w)

        return output