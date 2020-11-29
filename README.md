## General Point Sampling Package for Pytorch

This repository contains basic cuda implementation for pointwise sampling from 2D feature map, and also export APIs for operation with point-sampling as atomic operation (Resize, Affine, FlowWarp and ROI)

### Install

Install torch first, compile the source file and install it
```python3
python3 setup.py install
```

### Usage
You can use the interface in `ops.py` file or direct sampling function in `utils.py`
```python3

import grid_sample.ops as ops
import grid_sample.utils as utils

# use ops
func = ops.Affine()
out = func(tensor, affine_matrix)

# direct sampling
interp_type = 'bilinear'
out = utils.sample(tensor, points, interp_type)

```