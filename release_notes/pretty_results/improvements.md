* Implement torch.diagonal for named tensors (#30193) ([#30193](https://github.com/pytorch/pytorch/pull/30193)).
* Add list of supported XCode/CUDA versions to README ([#30407](https://github.com/pytorch/pytorch/pull/30407)).
* detect_anomaly() for SparseTensors (#29803) ([#29803](https://github.com/pytorch/pytorch/pull/29803)).
* Support 0-d tensors in CUDA MultiLabelMarginCriterion. (#30765) ([#30765](https://github.com/pytorch/pytorch/pull/30765)).
* Migrate max and min (binary) from TH to ATen. (#27185) ([#27185](https://github.com/pytorch/pytorch/pull/27185)).
* Enable len(dataloader) for iterable dataset (#23587) ([#23587](https://github.com/pytorch/pytorch/pull/23587)).
* add torch.square (#30719) ([#30719](https://github.com/pytorch/pytorch/pull/30719)).
* Add useful warnings for t.grad when it won't be populated for known reasons (#30531) ([#30531](https://github.com/pytorch/pytorch/pull/30531)).
* Add types for the remaining optimizers. (#31130) ([#31130](https://github.com/pytorch/pytorch/pull/31130)).
* Enabled roll for bool tensor (#31194) ([#31194](https://github.com/pytorch/pytorch/pull/31194)).
* Enable equality operator for bfloat16 CPU scalar types. (#30817) ([#30817](https://github.com/pytorch/pytorch/pull/30817)).
* Detect dill version in torch.save/load (#30985) ([#30985](https://github.com/pytorch/pytorch/pull/30985)).
* Enabled flip for bool tensors (#31267) ([#31267](https://github.com/pytorch/pytorch/pull/31267)).
* Added error message to indicate that reduction operations are not supported for dim>=64 (#31476) ([#31476](https://github.com/pytorch/pytorch/pull/31476)).
* Better error msg for autograd profiler + multi-worker dataloader crash (#31473) ([#31473](https://github.com/pytorch/pytorch/pull/31473)).
* add type promotion support for sparse tensors (#30429) ([#30429](https://github.com/pytorch/pytorch/pull/30429)).
* Raise warning for schedulers following chainable shedulers (#31125) ([#31125](https://github.com/pytorch/pytorch/pull/31125)).
* Conv transpose/backward split 32bit (#31510) ([#31510](https://github.com/pytorch/pytorch/pull/31510)).
* add additional types to indexing operations dispatch (#31692) ([#31692](https://github.com/pytorch/pytorch/pull/31692)).
* no_grad, enable_grad: support for decorating generator functions (#31792) ([#31792](https://github.com/pytorch/pytorch/pull/31792)).
* Add stub for transformer.py and MultiheadAttention Class. (#28396) ([#28396](https://github.com/pytorch/pytorch/pull/28396)).
* Use TORCH_CHECK instead of AT_ASSERT in torch::cuda::gather() (#27456) ([#27456](https://github.com/pytorch/pytorch/pull/27456)).
* named tensor max pooling support ([#31669](https://github.com/pytorch/pytorch/pull/31669)).
* Fix and add more padding mode support for Conv (#31784) ([#31784](https://github.com/pytorch/pytorch/pull/31784)).
* Make torch.backends.mkldnn usable without import ([#32055](https://github.com/pytorch/pytorch/pull/32055)).
* out variant for native_batch_norm forward (#29192) ([#29192](https://github.com/pytorch/pytorch/pull/29192)).
* support empty batch in group normalization (#32401) ([#32401](https://github.com/pytorch/pytorch/pull/32401)).
* add missing align_corners annotation (#32492) ([#32492](https://github.com/pytorch/pytorch/pull/32492)).
* Fix nll_loss to support empty tensors on GPU (#31491) ([#31491](https://github.com/pytorch/pytorch/pull/31491)).
* Support 3D attention mask in MultiheadAttention. (#31996) ([#31996](https://github.com/pytorch/pytorch/pull/31996)).
* Update linspace types (#32218) ([#32218](https://github.com/pytorch/pytorch/pull/32218)).
* 0-dim batch size input for interpolate. (#32400) ([#32400](https://github.com/pytorch/pytorch/pull/32400)).
* added exception args to the returned error message (#32693) ([#32693](https://github.com/pytorch/pytorch/pull/32693)).
* enable empty batch for all flavor of convolutions (#32709) ([#32709](https://github.com/pytorch/pytorch/pull/32709)).
* make tests for empty inputs check zero parameter grads (#32820) ([#32820](https://github.com/pytorch/pytorch/pull/32820)).
* Update type hints for torch.optim.optimizer.Optimizer (#32900) ([#32900](https://github.com/pytorch/pytorch/pull/32900)).
* add missing method annotations to torch.Tensor (#30576) ([#30576](https://github.com/pytorch/pytorch/pull/30576)).
* Add missing `default_collate` in dataloader.pyi ([#28935](https://github.com/pytorch/pytorch/pull/28935)).
* Tests for verifying behaviour of BatchNorm using 0-dim batch sizes. (#32384) ([#32384](https://github.com/pytorch/pytorch/pull/32384)).
* Backward operation of torch.eig for real eigenvalues (#33090) ([#33090](https://github.com/pytorch/pytorch/pull/33090)).
* Use int64 in pdist kernel to handle batches >= 46342 #30583 (#31593) ([#31593](https://github.com/pytorch/pytorch/pull/31593)).
* add missing default value for LRScheduler.step() (#32411) ([#32411](https://github.com/pytorch/pytorch/pull/32411)).
* Add a warning sign for anomaly detection (#33176) (#33239) ([#33239](https://github.com/pytorch/pytorch/pull/33239)).
* Optimize error checking in mvlgamma (#32665) ([#32665](https://github.com/pytorch/pytorch/pull/32665)).
* fix typing bug of LambdaLR.__init__ (#33271) ([#33271](https://github.com/pytorch/pytorch/pull/33271)).
* Add type annotation for bias in _ConvNd (#32885) ([#32885](https://github.com/pytorch/pytorch/pull/32885)).
* Add 64-bit indexing support to THC index reductions (#33405) ([#33405](https://github.com/pytorch/pytorch/pull/33405)).
* adding IterableDataset to utils.data.__init__ (#33543) ([#33543](https://github.com/pytorch/pytorch/pull/33543)).
* Add typing info for data members of utils.data.sampler classes (#33679) ([#33679](https://github.com/pytorch/pytorch/pull/33679)).
* Fix typing error of torch/nn/modules/container.pyi.in (#33686) ([#33686](https://github.com/pytorch/pytorch/pull/33686)).
* fix bugs in gen_pyi.py (#33748) ([#33748](https://github.com/pytorch/pytorch/pull/33748)).
* add bfloat16 conversion method in type stub (__init__.pyi) (#33747) ([#33747](https://github.com/pytorch/pytorch/pull/33747)).
* Better handing of Autograd+Fork errors. (#33885) ([#33885](https://github.com/pytorch/pytorch/pull/33885)).
* disable leaky_relu_ backward calculation with negative slope (#33639) ([#33639](https://github.com/pytorch/pytorch/pull/33639)).
* Improve dll loading logic on Windows (#33856) ([#33856](https://github.com/pytorch/pytorch/pull/33856)).
* Throw an error if nbytes is called on a sparse tensor. (#33897) ([#33897](https://github.com/pytorch/pytorch/pull/33897)).
* Enable Tensor.random_(from, to) for half on CPU (#34030) ([#34030](https://github.com/pytorch/pytorch/pull/34030)).
* Expose `CUDACachingAllocator` `raw_alloc` and `raw_delete` to python (#33860) ([#33860](https://github.com/pytorch/pytorch/pull/33860)).
* cuDNN convolution try multiple algo (#33073) ([#33073](https://github.com/pytorch/pytorch/pull/33073)).
* Fix doc and type hints for "torch.add"; fix deprecated python calls in tests (#33935) ([#33935](https://github.com/pytorch/pytorch/pull/33935)).
* Warns on read-only Numpy array->tensor conversion (#33615) ([#33615](https://github.com/pytorch/pytorch/pull/33615)).
* Fixed stub for AdamW (#34299) ([#34299](https://github.com/pytorch/pytorch/pull/34299)).
* Print the current Node name in anomaly mode (#33875) ([#33875](https://github.com/pytorch/pytorch/pull/33875)).
* Add Tensor overload for start in narrow. (#34317) ([#34317](https://github.com/pytorch/pytorch/pull/34317)).
* Support for Tensor Shape Type Hint (#34595) ([#34595](https://github.com/pytorch/pytorch/pull/34595)).
* Makes floor_divide a method, adds sparse floor division (#34552) ([#34552](https://github.com/pytorch/pytorch/pull/34552)).
* Check if rnn weights need to be flattened (#34265) ([#34265](https://github.com/pytorch/pytorch/pull/34265)).
* Revert D20312366: [pytorch][PR] Added type promotion logic for complex numbers ([#None](https://github.com/pytorch/pytorch/pull/None)).
* Warns when performing integer division with div and addcdiv (#34570) ([#34570](https://github.com/pytorch/pytorch/pull/34570)).
* Added type promotion logic for complex numbers (#34093) ([#34093](https://github.com/pytorch/pytorch/pull/34093)).
* Makes floor_divide a method, adds sparse floor division (#34552) ([#34552](https://github.com/pytorch/pytorch/pull/34552)).
* Revert D20497453: [pytorch][PR] Makes floor_divide a method, adds sparse floor division ([#None](https://github.com/pytorch/pytorch/pull/None)).
* fix type stub errors (#33762) ([#33762](https://github.com/pytorch/pytorch/pull/33762)).
* Adds true_divide function, analogous to Python 's, JAX's, NumPy's (true) division (#34236) ([#34236](https://github.com/pytorch/pytorch/pull/34236)).
* Add floor_divide function (#30493) ([#30493](https://github.com/pytorch/pytorch/pull/30493)).
