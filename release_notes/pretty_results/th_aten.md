* Port ELU activation to Aten (#29275) ([#29275](https://github.com/pytorch/pytorch/pull/29275)).
* Move rrelu to Aten(CPU) (#31094) ([#31094](https://github.com/pytorch/pytorch/pull/31094)).
* Port softplus activation to Aten(CPU+CUDA) (#30504) ([#30504](https://github.com/pytorch/pytorch/pull/30504)).
* Add op bitwise_or (#31559) ([#31559](https://github.com/pytorch/pytorch/pull/31559)).
* Move cauchy to Aten(CPU) (#31824) ([#31824](https://github.com/pytorch/pytorch/pull/31824)).
* Move lshift to Aten (#31566) ([#31566](https://github.com/pytorch/pytorch/pull/31566)).
* Move log_sigmoid to Aten(CPU) (#30958) ([#30958](https://github.com/pytorch/pytorch/pull/30958)).
* Move geometric to Aten(CPU) (#31878) ([#31878](https://github.com/pytorch/pytorch/pull/31878)).
* Move rshift to Aten (#31594) ([#31594](https://github.com/pytorch/pytorch/pull/31594)).
* porting scatter_add to ATen (CPU) (#31662) ([#31662](https://github.com/pytorch/pytorch/pull/31662)).
* Move log_normal to Aten(CPU) (#31854) ([#31854](https://github.com/pytorch/pytorch/pull/31854)).
* Migrate max and min (binary) from TH to ATen. (#30851) ([#30851](https://github.com/pytorch/pytorch/pull/30851)).
* porting gather to ATen using TensorIterator with multithreading support. (#32425) ([#32425](https://github.com/pytorch/pytorch/pull/32425)).
* Move exponential_ from TH to Aten (CPU) (#32501) ([#32501](https://github.com/pytorch/pytorch/pull/32501)).
* Port BCELoss to ATen to increase accuracy (#31365) ([#31365](https://github.com/pytorch/pytorch/pull/31365)).
* Vectorized memory access in TensorIterator GPU loop for 1d contiguous case (#32383) ([#32383](https://github.com/pytorch/pytorch/pull/32383)).
* Move normal distribution to Aten(CPU) (#32031) ([#32031](https://github.com/pytorch/pytorch/pull/32031)).
* [pytorch] Migrating index_add cuda to ATen (#30573) ([#30573](https://github.com/pytorch/pytorch/pull/30573)).
* Move where cuda implementation to TensorIterator (#32984) ([#32984](https://github.com/pytorch/pytorch/pull/32984)).
* Move where cuda implementation to TensorIterator (#33228) ([#33228](https://github.com/pytorch/pytorch/pull/33228)).
* Migrate dist from TH to ATen(CPU, CUDA) (#29714) ([#29714](https://github.com/pytorch/pytorch/pull/29714)).
* glu: port cpu forward implementation to ATen (#26410) ([#26410](https://github.com/pytorch/pytorch/pull/26410)).
* [WIP] migrate scatter_ to ATen CPU (+multithreading, nondeterministic) (#33139) ([#33139](https://github.com/pytorch/pytorch/pull/33139)).
* Migrate `random_` from the TH to Aten (CPU) (#32534) ([#32534](https://github.com/pytorch/pytorch/pull/32534)).
* Migrate _cat from TH to ATen (CUDA) (#33237) ([#33237](https://github.com/pytorch/pytorch/pull/33237)).
* Move cumprod and cumsum to Aten(CPU) (#33280) ([#33280](https://github.com/pytorch/pytorch/pull/33280)).
* Migrate `fmod` and `fmod_` from TH to ATen (CPU) (#33592) ([#33592](https://github.com/pytorch/pytorch/pull/33592)).
* port masked_fill from TH to ATen (#33330) ([#33330](https://github.com/pytorch/pytorch/pull/33330)).
* Migrate `random_` from the TH to Aten (CPU and CUDA) (#33663) ([#33663](https://github.com/pytorch/pytorch/pull/33663)).
* Migrate prelu from CUDA_tensor_apply2 to TensorIterator (#34003) ([#34003](https://github.com/pytorch/pytorch/pull/34003)).
* Migrate gamma grad from CUDA_tensor_apply3 to TensorIterator (#34020) ([#34020](https://github.com/pytorch/pytorch/pull/34020)).
* Migrate dirichlet from CUDA_tensor_apply3 to TensorIterator (#34021) ([#34021](https://github.com/pytorch/pytorch/pull/34021)).
* [RESUBMIT] [pytorch] Migrating index_add cuda to ATen (#33548) ([#33548](https://github.com/pytorch/pytorch/pull/33548)).
* Migrate kl_div_backward from CUDA_tensor_apply3 to TensorIterator (#34022) ([#34022](https://github.com/pytorch/pytorch/pull/34022)).
* Migrate Lerp from CUDA_tensor_apply4 to TensorIterator (#33994) ([#33994](https://github.com/pytorch/pytorch/pull/33994)).
* Migrate bce loss from CUDA_tensor_apply3 to TensorIterator (#34023) ([#34023](https://github.com/pytorch/pytorch/pull/34023)).
* Migrate lerp from CUDA_tensor_apply3 to TensorIterator (#34025) ([#34025](https://github.com/pytorch/pytorch/pull/34025)).
* [pytorch]Migrate _th_ger to Aten and kill resize_scalar in codegen (#33792) ([#33792](https://github.com/pytorch/pytorch/pull/33792)).
* Port `remainder` from TH to ATen (CPU and CUDA) (#34136) ([#34136](https://github.com/pytorch/pytorch/pull/34136)).
* Move min and max(reduce all) to Aten(CPU) (#33936) ([#33936](https://github.com/pytorch/pytorch/pull/33936)).
* Migrate binary_cross_entropy_backward from CUDA_tensor_apply4 to (#33995) ([#33995](https://github.com/pytorch/pytorch/pull/33995)).
* Migrate dirichlet_grad from CUDA_tensor_apply4 to TensorIterator (#33996) ([#33996](https://github.com/pytorch/pytorch/pull/33996)).
* Move glu to Aten(CPU) (#33179) ([#33179](https://github.com/pytorch/pytorch/pull/33179)).
