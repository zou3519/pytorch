* Force early initialization of OpenMP in forked children (#29006) ([#29006](https://github.com/pytorch/pytorch/pull/29006)).
* fix AvgPool2d for 2^31-1 sized inputs, and get test_cuda_kernel_loop_… (#30771) ([#30771](https://github.com/pytorch/pytorch/pull/30771)).
* Fix a CUDA memory leak in MultiLabelMarginCriterion error checking. (#30767) ([#30767](https://github.com/pytorch/pytorch/pull/30767)).
* Wrap warning handler in a function to avoid siof (#30800) ([#30800](https://github.com/pytorch/pytorch/pull/30800)).
* Fix `os.register_at_fork` not defined on Windows (#30809) ([#30809](https://github.com/pytorch/pytorch/pull/30809)).
* Fix error checking of CUDA multi_margin_loss. (#30825) ([#30825](https://github.com/pytorch/pytorch/pull/30825)).
* Fix half->float case of softmax backward when inner_size is not 1 (#30838) ([#30838](https://github.com/pytorch/pytorch/pull/30838)).
* Fix reading `__cuda_array_interface__` without strides (#24947) ([#24947](https://github.com/pytorch/pytorch/pull/24947)).
* Make nn.Module `forward()` type annotation more permissive (#31057) ([#31057](https://github.com/pytorch/pytorch/pull/31057)).
* Correctly handle scalar types, fix parse of numpy ints (#30486) ([#30486](https://github.com/pytorch/pytorch/pull/30486)).
* Refactor test for unique and unique_consecutive and fix some bugs (#31211) ([#31211](https://github.com/pytorch/pytorch/pull/31211)).
* fix view call on discontiguous tensor in to_sparse_backward (#31223) ([#31223](https://github.com/pytorch/pytorch/pull/31223)).
* Fix unflatten when dim is a negative integer (#31208) ([#31208](https://github.com/pytorch/pytorch/pull/31208)).
* caffe2::TypeInfo fix when using clang-cl on Windows (#31364) ([#31364](https://github.com/pytorch/pytorch/pull/31364)).
* check devices for all input tensors in index_put (#31280) ([#31280](https://github.com/pytorch/pytorch/pull/31280)).
* Don't dispatch to cudnn if it is not possible to make it 32bit by splitting batch dim (#31383) ([#31383](https://github.com/pytorch/pytorch/pull/31383)).
* Split on batch dimension when 32bit indexing not enough for convolution forward (#31379) ([#31379](https://github.com/pytorch/pytorch/pull/31379)).
* set stream everytime when we get a cuBlas handle (#31537) ([#31537](https://github.com/pytorch/pytorch/pull/31537)).
* set stream everytime when we get a cuSparse handle (#31538) ([#31538](https://github.com/pytorch/pytorch/pull/31538)).
* set stream everytime when we get a cuDNN handle (#31541) ([#31541](https://github.com/pytorch/pytorch/pull/31541)).
* Ensure legacy sparse constructor/new doesn't interpret python data as tensor data. (#31490) ([#31490](https://github.com/pytorch/pytorch/pull/31490)).
* updated code to ensure error check for negative dims ([#31636](https://github.com/pytorch/pytorch/pull/31636)).
* Fix nvcc math functions for MSVC 2019 (#31704) ([#31704](https://github.com/pytorch/pytorch/pull/31704)).
* Fix cumsum error for tensors with zero elements (#31694) ([#31694](https://github.com/pytorch/pytorch/pull/31694)).
* Raise ValueError if CUDA device is specified without specifying the : (#29087) ([#29087](https://github.com/pytorch/pytorch/pull/29087)).
* Ensure autograd callbacks are called only once for reentrant backward. (#31909) ([#31909](https://github.com/pytorch/pytorch/pull/31909)).
* Raise an error if torch.cat is given `out` as one of the input tensors (#30577) ([#30577](https://github.com/pytorch/pytorch/pull/30577)).
* Fix c10::util::get_fully_qualified_type_name for MSVC (#31313) ([#31313](https://github.com/pytorch/pytorch/pull/31313)).
* Avoid reference invalidation in cuda SpectralOps' plan_caches (#31861) ([#31861](https://github.com/pytorch/pytorch/pull/31861)).
* Ensure the original grad_mode is restored during backward (#31884) ([#31884](https://github.com/pytorch/pytorch/pull/31884)).
* Lock graph_task before writing leaf_streams. (#31995) ([#31995](https://github.com/pytorch/pytorch/pull/31995)).
* Fix tensordot allowing negative dims (#31954) ([#31954](https://github.com/pytorch/pytorch/pull/31954)).
* Fix cumprod error for tensors with zero elements (#32070) ([#32070](https://github.com/pytorch/pytorch/pull/32070)).
* Fix tensor^tensor derivative for 0 base entries ([#32062](https://github.com/pytorch/pytorch/pull/32062)).
* Fix scalar^tensor derivative for scalars that are zero ([#32063](https://github.com/pytorch/pytorch/pull/32063)).
* Fix weight backward for cudnn conv of large tensor (#31889) ([#31889](https://github.com/pytorch/pytorch/pull/31889)).
* Make type of `Tensor.type()` more specific (#32353) ([#32353](https://github.com/pytorch/pytorch/pull/32353)).
* Always return a new tensor from nn.functional.pad (#32350) ([#32350](https://github.com/pytorch/pytorch/pull/32350)).
* Fix race condition for to() backward that spans devices (#31930) ([#31930](https://github.com/pytorch/pytorch/pull/31930)).
* Raise error for code that risk deadlock (#32295) ([#32295](https://github.com/pytorch/pytorch/pull/32295)).
* fix view listing in autograd codegen (#32044) ([#32044](https://github.com/pytorch/pytorch/pull/32044)).
* Fix wrong typing (torch/nn/parameter.pyi) (#32617) ([#32617](https://github.com/pytorch/pytorch/pull/32617)).
* verify input sizes for instance norm and group norm (#29082) ([#29082](https://github.com/pytorch/pytorch/pull/29082)).
* Fixes moving after weight norm application (#32563) ([#32563](https://github.com/pytorch/pytorch/pull/32563)).
* div_kernel: throw when dividing by integer zero (#32629) ([#32629](https://github.com/pytorch/pytorch/pull/32629)).
* Make sure temporary vectors are properly initialized in avx2 code (#32722) ([#32722](https://github.com/pytorch/pytorch/pull/32722)).
* Fix crash of SobolEngine if default tensor type is cuda (#32496) ([#32496](https://github.com/pytorch/pytorch/pull/32496)).
* Logspace fixes (#32744) ([#32744](https://github.com/pytorch/pytorch/pull/32744)).
* Fixes moving after weight norm application (#32563) ([#32563](https://github.com/pytorch/pytorch/pull/32563)).
* Cudnn bn size fix (#32763) ([#32763](https://github.com/pytorch/pytorch/pull/32763)).
* Solves Issue #32750 - torch.prod now works fine with FP16 Input Tensor and FP32 Output Tensor (#32831) ([#32831](https://github.com/pytorch/pytorch/pull/32831)).
* Fix upsampling test case on ppc (#32786) ([#32786](https://github.com/pytorch/pytorch/pull/32786)).
* Enable MKL on MacOS if installed (#32905) ([#32905](https://github.com/pytorch/pytorch/pull/32905)).
* Properly handle NaN in binary max and min (#32541) ([#32541](https://github.com/pytorch/pytorch/pull/32541)).
* min, max: check that operand and outputs are on the same device type (#32862) ([#32862](https://github.com/pytorch/pytorch/pull/32862)).
* properly update _flat_weights in RNN modules (#32939) ([#32939](https://github.com/pytorch/pytorch/pull/32939)).
* Add size checks to `torch.stack` (#32931) ([#32931](https://github.com/pytorch/pytorch/pull/32931)).
* [aten] fix vector memory leak (#32478) ([#32478](https://github.com/pytorch/pytorch/pull/32478)).
* Properly update _flat_weights in RNN models (#32989) ([#32989](https://github.com/pytorch/pytorch/pull/32989)).
* fix #30480 torch.normal shape checking is broken (#32243) ([#32243](https://github.com/pytorch/pytorch/pull/32243)).
* Add more checks to custom Function (#33069) ([#33069](https://github.com/pytorch/pytorch/pull/33069)).
* Remove return value for __exit__ (#32997) ([#32997](https://github.com/pytorch/pytorch/pull/32997)).
* Add nice error message if missing overrides in custom autograd.Function ([#33142](https://github.com/pytorch/pytorch/pull/33142)).
* fix #30480 torch.normal shape checking is broken (#32243) (#33050) ([#33050](https://github.com/pytorch/pytorch/pull/33050)).
* TORCH_INTERNAL_ASSERT_DEBUG_ONLY not eating message string (#33251) ([#33251](https://github.com/pytorch/pytorch/pull/33251)).
* Updates numpy to tensor negative stride error message (#33254) ([#33254](https://github.com/pytorch/pytorch/pull/33254)).
* Fix index truncation in argmin/max for large tensors (#33310) ([#33310](https://github.com/pytorch/pytorch/pull/33310)).
* [pytorch] correct input size check for GroupNorm (#33008) ([#33008](https://github.com/pytorch/pytorch/pull/33008)).
* Dirac init compatibility with group convolutions (#32825) ([#32825](https://github.com/pytorch/pytorch/pull/32825)).
* Fix isnan for integral types in MSVC (#33483) ([#33483](https://github.com/pytorch/pytorch/pull/33483)).
* Fixes #33001 (#33456) ([#33456](https://github.com/pytorch/pytorch/pull/33456)).
* Fix LambdaLR scheduler side effects (#32848) ([#32848](https://github.com/pytorch/pytorch/pull/32848)).
* Add missing weight_decay parameter validation for Adam and AdamW (#33126) ([#33126](https://github.com/pytorch/pytorch/pull/33126)).
* Check for consistent devices in at::where (#33432) ([#33432](https://github.com/pytorch/pytorch/pull/33432)).
* Ensure that lambda is no less than zero in softshrink (#33201) ([#33201](https://github.com/pytorch/pytorch/pull/33201)).
* Fixes cuda->numpy and non-strided->numpy segfaults (#33612) ([#33612](https://github.com/pytorch/pytorch/pull/33612)).
* Fix NaN handling in torch.mv. (#31666) ([#31666](https://github.com/pytorch/pytorch/pull/31666)).
* Make ELU great again (#33244) ([#33244](https://github.com/pytorch/pytorch/pull/33244)).
* [pytorch] blas gemm fix for k=0 (#33419) ([#33419](https://github.com/pytorch/pytorch/pull/33419)).
* Fix potential hang when exiting main process (#33721) ([#33721](https://github.com/pytorch/pytorch/pull/33721)).
* Skip manual backward for `cdist` with case `p=2` (#31167) ([#31167](https://github.com/pytorch/pytorch/pull/31167)).
* Fix grid_sample gradients at image borders (#32829) ([#32829](https://github.com/pytorch/pytorch/pull/32829)).
* Avoid problematic pickle usages on Python 3.8.0 and 3.8.1 (#33824) ([#33824](https://github.com/pytorch/pytorch/pull/33824)).
* Fix index_put when tensor length > int_max (#33753) ([#33753](https://github.com/pytorch/pytorch/pull/33753)).
* prevent crash on exit from static destructor race (#33955) ([#33955](https://github.com/pytorch/pytorch/pull/33955)).
* [pytorch] blas gemm fix for k=0 (#33819) ([#33819](https://github.com/pytorch/pytorch/pull/33819)).
* Fix MKLDNN conv2d 5d weight handling (#34115) ([#34115](https://github.com/pytorch/pytorch/pull/34115)).
* Allow output to zero-strided tensors if the size is <= 1 along that dim (#34100) ([#34100](https://github.com/pytorch/pytorch/pull/34100)).
* Make sure Vec256 int32_t and int16_t loadu temprary arrays are properly initialized (#34281) ([#34281](https://github.com/pytorch/pytorch/pull/34281)).
* Fix Conv.cpp, &&= is not a C++ operator (#34381) ([#34381](https://github.com/pytorch/pytorch/pull/34381)).
* Fix the missing ';' in Conv.cpp (#34448) ([#34448](https://github.com/pytorch/pytorch/pull/34448)).
* Fix cudnn 64bit indexing issue (#34407) ([#34407](https://github.com/pytorch/pytorch/pull/34407)).
* Remove custom function in no_grad block error message (#33896) ([#33896](https://github.com/pytorch/pytorch/pull/33896)).
* Fix #33562 (uncaught domain_error on macOS) (#34301) ([#34301](https://github.com/pytorch/pytorch/pull/34301)).
* convert counter back to list #33229 (#33356) ([#33356](https://github.com/pytorch/pytorch/pull/33356)).
* Prohibit copying autograd engines (#34567) ([#34567](https://github.com/pytorch/pytorch/pull/34567)).
* Fix _cat operator (#34591) ([#34591](https://github.com/pytorch/pytorch/pull/34591)).
* Fix version check for grad_fn for views (#34145) ([#34145](https://github.com/pytorch/pytorch/pull/34145)).
* Remove hotpatches that circumvent MAGMA bug (#34357) ([#34357](https://github.com/pytorch/pytorch/pull/34357)).
* Fix bug in baddbmm corner case (#33467) (#33538) ([#33538](https://github.com/pytorch/pytorch/pull/33538)).
* solve conv3d backward get incorrect result problem (#34358) ([#34358](https://github.com/pytorch/pytorch/pull/34358)).
* Make autogen functions correct for multiple outputs and views (#31990) ([#31990](https://github.com/pytorch/pytorch/pull/31990)).
* Fix MagmaInitializesCorrectly_CUDA by using an invertible matrix (#32547) ([#32547](https://github.com/pytorch/pytorch/pull/32547)).
* Back out "Make autogen functions correct for multiple outputs and views" (#32681) ([#32681](https://github.com/pytorch/pytorch/pull/32681)).
* Add allow_rebase_history flag and fix codegen functions for multiple views (#32790) ([#32790](https://github.com/pytorch/pytorch/pull/32790)).
