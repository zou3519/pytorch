* Improve compare kernel (#29743) ([#29743](https://github.com/pytorch/pytorch/pull/29743)).
* improve .view() performance (#30554) ([#30554](https://github.com/pytorch/pytorch/pull/30554)).
* don't use size()/stride() functions in TensorImpl, use size_[d]/stride_[d] instead (#30452) ([#30452](https://github.com/pytorch/pytorch/pull/30452)).
* Move Tanh backward to Aten(CPU+CUDA) (#30224) ([#30224](https://github.com/pytorch/pytorch/pull/30224)).
* Vectorize smooth L1 loss backward function on CPU. (#30046) ([#30046](https://github.com/pytorch/pytorch/pull/30046)).
* Move hardtanh activation to Aten(CPU, CUDA) (#30152) ([#30152](https://github.com/pytorch/pytorch/pull/30152)).
* Reduce intrusive_ptr incref/decref costs (#30709) ([#30709](https://github.com/pytorch/pytorch/pull/30709)).
* embedding_bag make_bag_size optimization (#30701) ([#30701](https://github.com/pytorch/pytorch/pull/30701)).
* Move Softshrink activation to Aten(CPU+CUDA) (#30229) ([#30229](https://github.com/pytorch/pytorch/pull/30229)).
* Optimize LayerNorm with explicit vectorization using Vec256 (#29104) ([#29104](https://github.com/pytorch/pytorch/pull/29104)).
* Re-apply "[bert/RoBERTa] Optimize LayerNorm with explicit vectorization using Vec256" (#31127) ([#31127](https://github.com/pytorch/pytorch/pull/31127)).
* Fix copy kernel speed regression introduced in #29631 (#31279) ([#31279](https://github.com/pytorch/pytorch/pull/31279)).
* Move leaky_relu to Aten(CPU, CUDA) (#29899) ([#29899](https://github.com/pytorch/pytorch/pull/29899)).
* Migrate soft_margin_loss from the TH to Aten (CUDA+CPU) (#28135) ([#28135](https://github.com/pytorch/pytorch/pull/28135)).
* Add TORCH_DCHECK macro that checks only in debug builds (#31240) ([#31240](https://github.com/pytorch/pytorch/pull/31240)).
* Speed up `Tensor::has_names` for unnamed tensors (#31436) ([#31436](https://github.com/pytorch/pytorch/pull/31436)).
* optimize index_select performance on CPU with TensorIterator (#30598) ([#30598](https://github.com/pytorch/pytorch/pull/30598)).
* Don't handle bias inside cudnn_convolution* (#31524) ([#31524](https://github.com/pytorch/pytorch/pull/31524)).
* TensorIterator norm update (#31903) ([#31903](https://github.com/pytorch/pytorch/pull/31903)).
* Changed clip_grad_norm_ total_norm calculation (#32020) ([#32020](https://github.com/pytorch/pytorch/pull/32020)).
* Make an assert on a hotpath trigger only in DEBUG mode. (#32117) ([#32117](https://github.com/pytorch/pytorch/pull/32117)).
* TensorIterator unrolling and vectorized load - step 0, 1 (#31974) ([#31974](https://github.com/pytorch/pytorch/pull/31974)).
* F.normalize uses clamp_min_ inplace (#32360) ([#32360](https://github.com/pytorch/pytorch/pull/32360)).
* duplicate symbols with AT_PARALLEL_OPENMP=0 (#32568) ([#32568](https://github.com/pytorch/pytorch/pull/32568)).
* Improved speed of frobenous norm for non-complex dtype (#30871) ([#30871](https://github.com/pytorch/pytorch/pull/30871)).
* Refreshing numel on a stride update is pointless. (#32116) ([#32116](https://github.com/pytorch/pytorch/pull/32116)).
* Vectorize softplus and its backward function on CPU (#32944) ([#32944](https://github.com/pytorch/pytorch/pull/32944)).
* fix gather regression by not materializing loop vars in the error mes… (#33108) ([#33108](https://github.com/pytorch/pytorch/pull/33108)).
* Add zero_mask function for vectorized functions. (#32985) ([#32985](https://github.com/pytorch/pytorch/pull/32985)).
* optimize cat performance on CPU with TensorIterator (#30806) ([#30806](https://github.com/pytorch/pytorch/pull/30806)).
* bad tbb lambda capture, bad chunk size (#30352) ([#30352](https://github.com/pytorch/pytorch/pull/30352)).
* Opitmize Unfold3d to improve performance of Conv3d (#33191) ([#33191](https://github.com/pytorch/pytorch/pull/33191)).
* Workaround performance bug / memory leak in GOMP (#32875) ([#32875](https://github.com/pytorch/pytorch/pull/32875)).
* Vectorize elu and its backward function on CPU (#32986) ([#32986](https://github.com/pytorch/pytorch/pull/32986)).
* fast setup for output tensor in tensor iterator (#33165) ([#33165](https://github.com/pytorch/pytorch/pull/33165)).
* Optimize Unfold3dAcc to improve performance of conv3d backward (#33317) ([#33317](https://github.com/pytorch/pytorch/pull/33317)).
* Remove gpu_kernel_with_index (#33370) ([#33370](https://github.com/pytorch/pytorch/pull/33370)).
* Allow vectorized gpu loop to have different argument types (#33222) ([#33222](https://github.com/pytorch/pytorch/pull/33222)).
* Revert "Revert D19964089: [pytorch][PR] Allow vectorized gpu loop to … (#33553) ([#33553](https://github.com/pytorch/pytorch/pull/33553)).
* improve roll performance (#33623) ([#33623](https://github.com/pytorch/pytorch/pull/33623)).
* Bounds checking for functor execution in vectorized/unrolled kernels (#33642) ([#33642](https://github.com/pytorch/pytorch/pull/33642)).
* improve EmbeddingBag performance on cuda (#33589) ([#33589](https://github.com/pytorch/pytorch/pull/33589)).
* Fix torch.cat() performance regression on single core CPU (#33534) ([#33534](https://github.com/pytorch/pytorch/pull/33534)).
* Remove unnecessary tensor copies (#33732) ([#33732](https://github.com/pytorch/pytorch/pull/33732)).
* clang intrinsics targeting (#33958) ([#33958](https://github.com/pytorch/pytorch/pull/33958)).
* CUDA Vectorized Dropout (#33879) ([#33879](https://github.com/pytorch/pytorch/pull/33879)).
* TH: Defer to ATen's AVX detection code (#34088) ([#34088](https://github.com/pytorch/pytorch/pull/34088)).
* optimize UpSampleNearest 1d 2d and 3d performance on CPU (#31452) ([#31452](https://github.com/pytorch/pytorch/pull/31452)).
* Remove cudaMemcpy on full memory overlap (#34548) ([#34548](https://github.com/pytorch/pytorch/pull/34548)).
* CUDA Loops: move address computation into policy, make policy.load load all arguments (#33720) ([#33720](https://github.com/pytorch/pytorch/pull/33720)).
* improve batch_norm contiguous case's performance (#34530) ([#34530](https://github.com/pytorch/pytorch/pull/34530)).
* Avoid clone for sparse tensors during accumulation of grads. (#33427) ([#33427](https://github.com/pytorch/pytorch/pull/33427)).
* [pytorch][embeddingbag] Parallelize the EmbeddingBag operator (#4049) ([#27477](https://github.com/pytorch/pytorch/pull/27477)).
