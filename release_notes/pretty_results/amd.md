* Ensure MIOpen is called on same stream as operator for RNN (#30672) ([#30672](https://github.com/pytorch/pytorch/pull/30672)).
* Check the error return of nvrtcGetProgramLogSize and nvrtcGetProgramLog (#30663) ([#30663](https://github.com/pytorch/pytorch/pull/30663)).
* Provide full path for buck hipification (#30746) ([#30746](https://github.com/pytorch/pytorch/pull/30746)).
* Fix 'initialized after field' error (#30908) ([#30908](https://github.com/pytorch/pytorch/pull/30908)).
* Small fixes for hipification (#31200) ([#31200](https://github.com/pytorch/pytorch/pull/31200)).
* Wraps assert statements in cuda kernels (#31276) ([#31276](https://github.com/pytorch/pytorch/pull/31276)).
* Replace assert with cuda assert macro (#31297) ([#31297](https://github.com/pytorch/pytorch/pull/31297)).
* Enable BFloat16 support for Convolutions on ROCm (#30948) ([#30948](https://github.com/pytorch/pytorch/pull/30948)).
* Support PyTorch ROCm CI on Ubuntu18.04 (#31886) ([#31886](https://github.com/pytorch/pytorch/pull/31886)).
* Abstract atomic add calls (#31992) ([#31992](https://github.com/pytorch/pytorch/pull/31992)).
* Revert "Support PyTorch ROCm CI on Ubuntu18.04 (#31886)" (#31946) ([#31946](https://github.com/pytorch/pytorch/pull/31946)).
* Install complete set of headers for ROCm build (#32076) ([#32076](https://github.com/pytorch/pytorch/pull/32076)).
* [ROCm] Adjust elementwise_kernel settings on ROCm (#32609) ([#32609](https://github.com/pytorch/pytorch/pull/32609)).
* Use C10_WARP_SIZE to fix functionality on HIP vs CUDA for batch_norm_backward_reduce (#33098) ([#33098](https://github.com/pytorch/pytorch/pull/33098)).
* [ROCm] Enable Bfloat16 type for activation and batch-norm ([#32065](https://github.com/pytorch/pytorch/pull/32065)).
* Disable flaky tests test_DistributedDataParallel and test_backend_group for ROCm (#33211) ([#33211](https://github.com/pytorch/pytorch/pull/33211)).
* Add ability to enable/disable MIOpen at runtime (#33118) ([#33118](https://github.com/pytorch/pytorch/pull/33118)).
* [ROCm] Added support for pytorch extensions to use HIP (#32669) ([#32669](https://github.com/pytorch/pytorch/pull/32669)).
* [ROCm] Enable BFloat16 type for pooling ops (#34166) ([#34166](https://github.com/pytorch/pytorch/pull/34166)).
* [ROCm] Enable double __shfl_down (#34103) ([#34103](https://github.com/pytorch/pytorch/pull/34103)).
* [ROCm] Enable BFloat16 type for loss functions and few misc ops required for resnet50 (#34469) ([#34469](https://github.com/pytorch/pytorch/pull/34469)).
* [ci] try to fix rocm builds (#34600) ([#34600](https://github.com/pytorch/pytorch/pull/34600)).
* [ROCm] Enable BFloat16 type for EmbeddingBag ops et al (#34630) ([#34630](https://github.com/pytorch/pytorch/pull/34630)).
* [ROCm] Fix for std::isnan regression in ROCm (#34664) ([#34664](https://github.com/pytorch/pytorch/pull/34664)).
* [ROCm] Enable Caffe2 video operators for ROCm ([#32610](https://github.com/pytorch/pytorch/pull/32610)).
* Enabling the nccl/rccl test for ROCM environment (#32340) ([#32340](https://github.com/pytorch/pytorch/pull/32340)).
* [ROCm] Enable 3D batch norms through MIOpen (#33262) ([#33262](https://github.com/pytorch/pytorch/pull/33262)).
* [ROCm] Enable 3D convolutions through ROCm (#33067) ([#33067](https://github.com/pytorch/pytorch/pull/33067)).
* [AMD] Remove num_gpu check for remote execution (#34318) ([#34318](https://github.com/pytorch/pytorch/pull/34318)).
* Enable test_distributed for ROCm but only with nccl backend [REDUX] (#32551) ([#32551](https://github.com/pytorch/pytorch/pull/32551)).
* Workaround hcc bug regarding extern "C" definitions (#30313) ([#30313](https://github.com/pytorch/pytorch/pull/30313)).
* Add native/quantized to the list of header rewrites (#31151) ([#31151](https://github.com/pytorch/pytorch/pull/31151)).
* Add tanh to c10::cuda::compat (#31844) ([#31844](https://github.com/pytorch/pytorch/pull/31844)).
* Set USE_RCCL cmake option (dependent on USE_NCCL) (#31341) ([#31341](https://github.com/pytorch/pytorch/pull/31341)).
