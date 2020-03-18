* Adding function to convert Module to channels last ([#28991](https://github.com/pytorch/pytorch/pull/28991)).
* Switch default memory format of _like operators to Preserve ([#30087](https://github.com/pytorch/pytorch/pull/30087)).
* Switch default memory format of to (and similar) operators to Preserve ([#30088](https://github.com/pytorch/pytorch/pull/30088)).
* Switch default memory format of clone operator to Preserve ([#30089](https://github.com/pytorch/pytorch/pull/30089)).
* cudnn grouped convolution nhwc patch (#31444) ([#31444](https://github.com/pytorch/pytorch/pull/31444)).
* Fix cudnn channels_last descriptors problem (#31952) ([#31952](https://github.com/pytorch/pytorch/pull/31952)).
* [NHWC CUDNN CONV]Update cudnn convolution memory_format behavior (#32482) ([#32482](https://github.com/pytorch/pytorch/pull/32482)).
* Expose Channel Last 3d enum ([#32947](https://github.com/pytorch/pytorch/pull/32947)).
* [CUDNN NHWC CONVOLUTION] Re-stride input tensors for wgrad in cudnn_convolution (#33784) ([#33784](https://github.com/pytorch/pytorch/pull/33784)).
* Clean warning message (#34143) ([#34143](https://github.com/pytorch/pytorch/pull/34143)).
* ChannelsLast3d support is_contiguous, contiguous, suggest_memory_format, caching (#33033) ([#33033](https://github.com/pytorch/pytorch/pull/33033)).
* Add nhwc memory format test for dropout (#34379) ([#34379](https://github.com/pytorch/pytorch/pull/34379)).
* Preserve memory format for torch.cat on CUDA (#34526) ([#34526](https://github.com/pytorch/pytorch/pull/34526)).
* Fix max_pool2d NHWC for large tensors; fix incorrect use of cudaGetLastError() (#34519) ([#34519](https://github.com/pytorch/pytorch/pull/34519)).
* Implement channels last upsample2d/3d forward pass kernel. (#34597) ([#34597](https://github.com/pytorch/pytorch/pull/34597)).
* Fix problem in NHWC max_pool2d; use accumulate type in NHWC max_pool2d (#34934) ([#34934](https://github.com/pytorch/pytorch/pull/34934)).
* Add MemoryFormat to TensorOptions, but not codegen. (#33704) ([#33704](https://github.com/pytorch/pytorch/pull/33704)).
