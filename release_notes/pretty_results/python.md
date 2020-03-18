* Adding elementwise kernel also operating on index (#28175) ([#28175](https://github.com/pytorch/pytorch/pull/28175)).
* Add torch.multiprocessing.create_processes ([#28493](https://github.com/pytorch/pytorch/pull/28493)).
* Fully deprecate variadic inputs of checkpoint_sequential (#25985) ([#25985](https://github.com/pytorch/pytorch/pull/25985)).
* Revert "Default to not build Caffe2 operators on Windows. (#29061)" (#30740) ([#30740](https://github.com/pytorch/pytorch/pull/30740)).
* Notify other threads before running callbacks (#31713) ([#31713](https://github.com/pytorch/pytorch/pull/31713)).
* Back out "fix view listing in autograd codegen" (#32720) ([#32720](https://github.com/pytorch/pytorch/pull/32720)).
* Fix for MKL detection script on Windows (#32970) ([#32970](https://github.com/pytorch/pytorch/pull/32970)).
* let user specify CUDA_HOST_COMPILER ([#32904](https://github.com/pytorch/pytorch/pull/32904)).
* Fix dispatch of argmax/argmin. (#32961) ([#32961](https://github.com/pytorch/pytorch/pull/32961)).
* Consider hub_dir alongside TORCH_HOME env variable for storing hub models (#32844) ([#32844](https://github.com/pytorch/pytorch/pull/32844)).
* Revert "Revert "Revert D19975411: Remove special case codegen for tril_indices/triu_indices." (#33572)" (#33742) ([#33742](https://github.com/pytorch/pytorch/pull/33742)).
* Try fix XLAPreAutograd with *_like functions. (#33848) ([#33848](https://github.com/pytorch/pytorch/pull/33848)).
* Pass all ops to XLA with additional info about whether it's compound (#33908) ([#33908](https://github.com/pytorch/pytorch/pull/33908)).
* Show errors from the tasks in the thread pool (#33938) ([#33938](https://github.com/pytorch/pytorch/pull/33938)).
* Add the build for runtime dispatch for AVX, AVX2 instruction set (#26125) ([#26125](https://github.com/pytorch/pytorch/pull/26125)).
* Adjust ProtoBufPatch to protobuf-3.11.x (#35008) ([#35008](https://github.com/pytorch/pytorch/pull/35008)).
* Raise error if a block can not be found from a CUDA tensor (#30870) ([#30870](https://github.com/pytorch/pytorch/pull/30870)).
