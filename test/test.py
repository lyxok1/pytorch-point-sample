import torch
import cv2
import numpy as np
import grid_sample.ops as ops
import grid_sample.utils as utils

def write_tensor(tensor, name):

    img = tensor.squeeze(0).permute(1, 2, 0).contiguous()
    img = img.cpu().numpy().astype(np.uint8)

    cv2.imwrite('{}.jpg'.format(name), img)

if __name__ == '__main__':

    img = cv2.imread('test.jpg').astype(np.float32)
    img = torch.from_numpy(img).cuda()
    h, w, c = img.shape

    img = img.permute(2, 0, 1).contiguous().unsqueeze(0)

    resize = ops.Resize(512)
    affine = ops.Affine()
    flowp = ops.WarpFlow()

    warp_matrix = torch.Tensor(
        [[
            [np.cos(np.pi / 3), np.sin(np.pi / 3), 0],
            [-np.sin(np.pi / 3), np.cos(np.pi / 3), -2],
        ]]
        ).cuda()

    flow = torch.zeros(1, 2, h, w).cuda()
    half_h, half_w = h // 2, w // 2
    flow[0, 1, :half_h, :] = 5
    flow[0, 1, half_h:, :] = -5
    flow[0, 0, :, :half_w] = 5
    flow[0, 0, :, half_w:] = -5

    resize_out = resize(img)
    affine_out = affine(img, warp_matrix)
    flow_out = flowp(img, flow)

    write_tensor(resize_out, 'resize')
    write_tensor(affine_out, 'affine')
    write_tensor(flow_out, 'flow')


