* Add inferBoundShapeOp ([#30101](https://github.com/pytorch/pytorch/pull/30101)).
* Support in-place update in IndexHashOp (#30275) ([#30275](https://github.com/pytorch/pytorch/pull/30275)).
* replace the SLSRQ for their right emulations in the replayer test (#30367) ([#30367](https://github.com/pytorch/pytorch/pull/30367)).
* Change logging to remove the word "error" from info log ([#30468](https://github.com/pytorch/pytorch/pull/30468)).
* make the order btw div and mul in adagrad update consistent (#30449) ([#30449](https://github.com/pytorch/pytorch/pull/30449)).
* move MaskedAdagrad to caffe2/operators/experimental/optimizers (#30714) ([#30714](https://github.com/pytorch/pytorch/pull/30714)).
* Back out "make the order btw div and mul in adagrad update consistent" (#30737) ([#30737](https://github.com/pytorch/pytorch/pull/30737)).
* call fp16<->fp32 routines in fbgemm from Half2Float and Float2Half operators (#30715) ([#30715](https://github.com/pytorch/pytorch/pull/30715)).
* Change interface from map of TensorShape to shapeInfoMap (#30802) ([#30802](https://github.com/pytorch/pytorch/pull/30802)).
* Automatic update of fbcode/onnx to c08a7b76cf7c1555ae37186f12be4d62b2c39b3b (#30619) ([#30619](https://github.com/pytorch/pytorch/pull/30619)).
* FCTransposed to FbFCPacked (#29766) ([#29766](https://github.com/pytorch/pytorch/pull/29766)).
* Support loading by blob in predictor ([#30805](https://github.com/pytorch/pytorch/pull/30805)).
* Calling JITed 8 Bit Fused SLS in FBGEMM from C2 (#30926) ([#30926](https://github.com/pytorch/pytorch/pull/30926)).
* add ZERO_COLLISION_HASH to caffe2 data type (#30912) ([#30912](https://github.com/pytorch/pytorch/pull/30912)).
* optimize MulGradient for common shapes (#19705) ([#19705](https://github.com/pytorch/pytorch/pull/19705)).
* Use AVX2 to increase frequency for FP16<->FP32 Caffe2 ops (#31203) ([#31203](https://github.com/pytorch/pytorch/pull/31203)).
* move AliasWithNameOp to caffe2/operators ([#31281](https://github.com/pytorch/pytorch/pull/31281)).
* move BatchPermutationOp to caffe2/operators ([#31350](https://github.com/pytorch/pytorch/pull/31350)).
* caffe2/event: allow multiple errors such as when cancelled (#31335) ([#31335](https://github.com/pytorch/pytorch/pull/31335)).
* fix zero-batch handling in convtranspose (#24341) ([#24341](https://github.com/pytorch/pytorch/pull/24341)).
* modify model to enable loading by blob (#31507) ([#31507](https://github.com/pytorch/pytorch/pull/31507)).
* Fix input_channels divisibility check in concat_split_op (#31448) ([#31448](https://github.com/pytorch/pytorch/pull/31448)).
* optimize FloatToFused8BitRowwiseQuantized and Fused8BitRowwiseQuantizedToFloat (#31470) ([#31470](https://github.com/pytorch/pytorch/pull/31470)).
* revert D18805532 and make numerics of masked adagrad consistent with unmasked adagrad (#30784) ([#30784](https://github.com/pytorch/pytorch/pull/30784)).
* Create byte-aware word lstm benchmark (#31260) ([#31260](https://github.com/pytorch/pytorch/pull/31260)).
* Integrating MaskedAdagrad ([#31640](https://github.com/pytorch/pytorch/pull/31640)).
* Optimize zero length input (#31602) ([#31602](https://github.com/pytorch/pytorch/pull/31602)).
* separate op for rowwise counter (#31612) ([#31612](https://github.com/pytorch/pytorch/pull/31612)).
* Integrate masked sparse Adagrad (#31641) ([#31641](https://github.com/pytorch/pytorch/pull/31641)).
* Change PackSegments to ensure consistent behavior between CPU and GPU ([#31673](https://github.com/pytorch/pytorch/pull/31673)).
* Allow to pass in masks through db (#31676) ([#31676](https://github.com/pytorch/pytorch/pull/31676)).
* Remove duplicated Numa detection code. (#30628) ([#30628](https://github.com/pytorch/pytorch/pull/30628)).
* Do not register `const float *` type on utiliy_ops.cu (#31583) ([#31583](https://github.com/pytorch/pytorch/pull/31583)).
* Update the descriptive error message for enforce fail (#31575) ([#31575](https://github.com/pytorch/pytorch/pull/31575)).
* Enable personalized FC weight_init and sparse_emb weight_init (#31707) ([#31707](https://github.com/pytorch/pytorch/pull/31707)).
* Fix -Wundef warning in conversions.h ([#31911](https://github.com/pytorch/pytorch/pull/31911)).
* Use simd version for fp16 conversions (#31897) ([#31897](https://github.com/pytorch/pytorch/pull/31897)).
* Combine all the user inputs together and convert them to fp16 (#31898) ([#31898](https://github.com/pytorch/pytorch/pull/31898)).
* add conversion functions to embedding tables (#31083) ([#31083](https://github.com/pytorch/pytorch/pull/31083)).
* Scale init for batch-norm and layer-norm (#31983) ([#31983](https://github.com/pytorch/pytorch/pull/31983)).
* Automatic update of fbcode/onnx to 65020daafa9183c769938b4512ce543fd5740f8f (#32125) ([#32125](https://github.com/pytorch/pytorch/pull/32125)).
* [caffe2] fix how np.clip is used in lengths_reducer_fused_{4,8}_rowwise_ops_test (#32086) ([#32086](https://github.com/pytorch/pytorch/pull/32086)).
* [Python] Deprecate use of scipy.misc.logsumexp and scipy.misc.comb (#32209) ([#32209](https://github.com/pytorch/pytorch/pull/32209)).
* Fix simple typo: whos -> whose (#31288) ([#31288](https://github.com/pytorch/pytorch/pull/31288)).
* Add CAFFE2_API to video decoding functions (#31187) ([#31187](https://github.com/pytorch/pytorch/pull/31187)).
* Support shape inference and lowering of SparseLengthsWeightedSumFused4BitRowwise (#32257) ([#32257](https://github.com/pytorch/pytorch/pull/32257)).
* Logical condition reduction (#32201) ([#32201](https://github.com/pytorch/pytorch/pull/32201)).
* Register RoIAlignRotated with C10 ([#30785](https://github.com/pytorch/pytorch/pull/30785)).
* fix spelling mistake: excpected -> expected ([#28817](https://github.com/pytorch/pytorch/pull/28817)).
* Add 64bit atomic fetch add (#32354) ([#32354](https://github.com/pytorch/pytorch/pull/32354)).
* C++ C2/Glow operator unittest ([#32258](https://github.com/pytorch/pytorch/pull/32258)).
* exposing CPU/GPU Copy ops (#32248) ([#32248](https://github.com/pytorch/pytorch/pull/32248)).
* Back out "Calling JITed 8 Bit Fused SLS in FBGEMM from C2" (#32381) ([#32381](https://github.com/pytorch/pytorch/pull/32381)).
* Redundant condition (#32396) ([#32396](https://github.com/pytorch/pytorch/pull/32396)).
* [caffe2] remove unnecessary np.set_printoptions and fix test errors (#32475) ([#32475](https://github.com/pytorch/pytorch/pull/32475)).
* [Rowwise Pruning][c2 op] Add Quantile Op (#32448) ([#32448](https://github.com/pytorch/pytorch/pull/32448)).
* [caffe2] use 2-stage EmbeddingSpMDM interface (#32271) ([#32271](https://github.com/pytorch/pytorch/pull/32271)).
* [caffe2] use JIT'ed fp32 SLS (#32413) ([#32413](https://github.com/pytorch/pytorch/pull/32413)).
* Enable mkldnn on windows (#31355) ([#31355](https://github.com/pytorch/pytorch/pull/31355)).
* Back out "Use simd version for fp16 conversions" (#32640) ([#32640](https://github.com/pytorch/pytorch/pull/32640)).
* Minor refactoring to improve code reuse (#32675) ([#32675](https://github.com/pytorch/pytorch/pull/32675)).
* Back out "[caffe2] use JIT'ed fp32 SLS" (#32711) ([#32711](https://github.com/pytorch/pytorch/pull/32711)).
* [caffe2] Early error throwing for currupted embeddings ([#32717](https://github.com/pytorch/pytorch/pull/32717)).
* Fix TensorProtosDBInput AttributeError (#32274) ([#32274](https://github.com/pytorch/pytorch/pull/32274)).
* [pytorch][embeddingbag_8bit] Add include_last_offset option to Fused 8bit EmbeddingBag and parallelize the op (#32683) ([#32683](https://github.com/pytorch/pytorch/pull/32683)).
* Py2 -> py3 for caffe2/caffe2/contrib/tensorboard (#32882) ([#32882](https://github.com/pytorch/pytorch/pull/32882)).
* Add a loop test for onnxified net (#32935) ([#32935](https://github.com/pytorch/pytorch/pull/32935)).
* Fix confusing "does not have GPU support" warning message (#30721) ([#30721](https://github.com/pytorch/pytorch/pull/30721)).
* [C2] Introduce extra_info force CPU tags for auto-generated iteration counter blobs (#32607) ([#32607](https://github.com/pytorch/pytorch/pull/32607)).
* Automatic update of fbcode/onnx to 8b3f7e2e7a0f2aba0e629e23d89f07c7fc0e6a5e (#33075) ([#33075](https://github.com/pytorch/pytorch/pull/33075)).
* [1/3] Bind IndexHash to PyTorch (#33015) ([#33015](https://github.com/pytorch/pytorch/pull/33015)).
* [caffe2] use JIT'ed fp32 SLS (#33123) ([#33123](https://github.com/pytorch/pytorch/pull/33123)).
* [2/3] Bind Bucketize to PyTorch (#33014) ([#33014](https://github.com/pytorch/pytorch/pull/33014)).
* [caffe2] remove dnnlowp log code (#33184) ([#33184](https://github.com/pytorch/pytorch/pull/33184)).
* [TVM] Add ReplaceNaN op (#33256) ([#33256](https://github.com/pytorch/pytorch/pull/33256)).
* [TVM] Add clip op to c2_frontend (#33257) ([#33257](https://github.com/pytorch/pytorch/pull/33257)).
* Fix compilation error when buildng with FFMPEG (#27589) ([#27589](https://github.com/pytorch/pytorch/pull/27589)).
* Automatic update of fbcode/onnx to 04a29addfd5b912812addb8dea5f8763fbfaad01 (#33328) ([#33328](https://github.com/pytorch/pytorch/pull/33328)).
* [caffe2] use JIT'ed fp16 SLS (#32432) ([#32432](https://github.com/pytorch/pytorch/pull/32432)).
* Suport all length one SLS op lowering: C2 part (#33332) ([#33332](https://github.com/pytorch/pytorch/pull/33332)).
* [caffe2] make order btw div and mul in adgrad consistent (#32974) ([#32974](https://github.com/pytorch/pytorch/pull/32974)).
* Remove unused variable (#33484) ([#33484](https://github.com/pytorch/pytorch/pull/33484)).
* [caffe2] simplify relative error expr (#32999) ([#32999](https://github.com/pytorch/pytorch/pull/32999)).
* [Caffe2][ThreadPool] Make sure numThreads does not exceed the number of big cores (#33523) ([#33523](https://github.com/pytorch/pytorch/pull/33523)).
* [caffe2] make fused rowwise quant/dequant op work for N-dim tensors (#33426) ([#33426](https://github.com/pytorch/pytorch/pull/33426)).
* [caffe2] fix invalid % escape in inline assembly strings (#33554) ([#33554](https://github.com/pytorch/pytorch/pull/33554)).
* [caffe2] use Clang identification macro in various places (#33574) ([#33574](https://github.com/pytorch/pytorch/pull/33574)).
* [TVM] Remove dynamic batch size dispatching (#33584) ([#33584](https://github.com/pytorch/pytorch/pull/33584)).
* [C2] Native GPU implementation for bucketize (#33529) ([#33529](https://github.com/pytorch/pytorch/pull/33529)).
* [caffe2] Add embedding empty ratio checker (disabled by default) (#33145) ([#33145](https://github.com/pytorch/pytorch/pull/33145)).
* [C2] Small improvement for elementwise_mul operator. (#33537) ([#33537](https://github.com/pytorch/pytorch/pull/33537)).
* [C2] Tiny changes to adagrad to make it slightly better. (#33727) ([#33727](https://github.com/pytorch/pytorch/pull/33727)).
* [Caffe2] Fix shape inference for element-wise operators (#33431) ([#33431](https://github.com/pytorch/pytorch/pull/33431)).
* Add partition info message to NetDef (#33616) ([#33616](https://github.com/pytorch/pytorch/pull/33616)).
* [caffe2] simplify caffe2 code with fbgemm handling block size 1 emb (#33774) ([#33774](https://github.com/pytorch/pytorch/pull/33774)).
* [caffe2] fix no return statement in constexpr function Clang error in TypeIndex.h (#33576) ([#33576](https://github.com/pytorch/pytorch/pull/33576)).
* [caffe2] fix no matching function min/max Clang errors (#33563) ([#33563](https://github.com/pytorch/pytorch/pull/33563)).
* update mapping of fake operators (#33946) ([#33946](https://github.com/pytorch/pytorch/pull/33946)).
* [caffe2] fix field initialization after base Clang errors (#33556) ([#33556](https://github.com/pytorch/pytorch/pull/33556)).
* [caffe2] fix atomicAdd redeclaration Clang error (#33559) ([#33559](https://github.com/pytorch/pytorch/pull/33559)).
* Fixing pthreadpool symbol conflict issue. (#33869) ([#33869](https://github.com/pytorch/pytorch/pull/33869)).
* [caffe2] Remove python2 from operator_test (#33977) ([#33977](https://github.com/pytorch/pytorch/pull/33977)).
* fix windows clang attributes (#33959) ([#33959](https://github.com/pytorch/pytorch/pull/33959)).
* Allow checking for cached module before asserting (#33954) ([#33954](https://github.com/pytorch/pytorch/pull/33954)).
* blacklist spatialBN until bitwise matching (#34092) ([#34092](https://github.com/pytorch/pytorch/pull/34092)).
* fix warnings reported by PVS (#33868) ([#33868](https://github.com/pytorch/pytorch/pull/33868)).
* [C2] Fix slowness of the ReshapeOp. (#33729) ([#33729](https://github.com/pytorch/pytorch/pull/33729)).
* fixup unit tests (#34105) ([#34105](https://github.com/pytorch/pytorch/pull/34105)).
* [caffe2] std::numeric_limits<double>::quiet_NaN() use instead of ::nan("") (#33566) ([#33566](https://github.com/pytorch/pytorch/pull/33566)).
* [caffe2] fix ambiguous call to 'fmaxType' THCHalfAutoNumerics.cuh (#33569) ([#33569](https://github.com/pytorch/pytorch/pull/33569)).
* NNPI op mapping correct SpatialBN NNPI op name (#34176) ([#34176](https://github.com/pytorch/pytorch/pull/34176)).
* Add backward Int8Quantize shape inference (#34152) ([#34152](https://github.com/pytorch/pytorch/pull/34152)).
* [caffe2] Fix signed unsigned comparison warning (#34161) ([#34161](https://github.com/pytorch/pytorch/pull/34161)).
* Added nullptr check for pthradpool_get_threads_count (#34087) ([#34087](https://github.com/pytorch/pytorch/pull/34087)).
* Allow converting IValue to vector<string> (#34269) ([#34269](https://github.com/pytorch/pytorch/pull/34269)).
* [ModelLoading] Use byte encoding for uint8, fp16 etc. instead of int32 (#34343) ([#34343](https://github.com/pytorch/pytorch/pull/34343)).
* [net_runner] Get shape info from qtensors (#34321) ([#34321](https://github.com/pytorch/pytorch/pull/34321)).
* [caffe2] do not declare __assert_fail in clang builds (#33893) ([#33893](https://github.com/pytorch/pytorch/pull/33893)).
* [net_transform] only skip ConstantFill for autogen_grad (#34628) ([#34628](https://github.com/pytorch/pytorch/pull/34628)).
* [DPER3] Blob Reorder (#33579) ([#33579](https://github.com/pytorch/pytorch/pull/33579)).
* [caffe2] Refactor out common util functions from tvm_transformer (#34652) ([#34652](https://github.com/pytorch/pytorch/pull/34652)).
* [DPER3][Shape Inference] Initial Shape Inference in DPER3 frontend (#33607) ([#33607](https://github.com/pytorch/pytorch/pull/33607)).
* [Shape Inference] Update shape inference in dper3 backend - C2 part (#34474) ([#34474](https://github.com/pytorch/pytorch/pull/34474)).
* [1/n][multi-tower] add partition info in predictor construction (#34175) ([#34175](https://github.com/pytorch/pytorch/pull/34175)).
* Export roi_align_gradient_op to c10 (#34776) ([#34776](https://github.com/pytorch/pytorch/pull/34776)).
* [fix][tiny][caffe2] Avoid triggering errors when allow ratio is 100% (#34757) ([#34757](https://github.com/pytorch/pytorch/pull/34757)).
* caffe2::OperatorBase do not need to be aware of at::Tensor functions (#34810) ([#34810](https://github.com/pytorch/pytorch/pull/34810)).
* [caffe2] fix Transpose2D calls in NHWC<->NCHW (#34625) ([#34625](https://github.com/pytorch/pytorch/pull/34625)).
* [Caffe2] Move more method implementations from tensor.h to tensor.cc (#34811) ([#34811](https://github.com/pytorch/pytorch/pull/34811)).
* [caffe2] open source 2/4-bit SLS operators (#34783) ([#34783](https://github.com/pytorch/pytorch/pull/34783)).
* Fix CMake Dev warning in caffe2/CMakeLists.txt (#34886) ([#34886](https://github.com/pytorch/pytorch/pull/34886)).
* move emulation libraries to contrib (#34861) ([#34861](https://github.com/pytorch/pytorch/pull/34861)).
* Support RowWiseSparseAdam on GPU (#34341) ([#34341](https://github.com/pytorch/pytorch/pull/34341)).
* [caffe2] open source 2/4-bit SLS operators (#34903) ([#34903](https://github.com/pytorch/pytorch/pull/34903)).
* Do not throw from CUDAContext destructor (#34756) ([#34756](https://github.com/pytorch/pytorch/pull/34756)).
* Bug fix of the histogram observers (#30970) ([#30970](https://github.com/pytorch/pytorch/pull/30970)).
* Fix 'template' keyword warning with clang-cl and clang.exe (#32104) ([#32104](https://github.com/pytorch/pytorch/pull/32104)).
* Bug fix of norm minimization for dev mode (#31462) ([#31462](https://github.com/pytorch/pytorch/pull/31462)).
* Add utils to inspect fp16/int8 packed weights (#32979) ([#32979](https://github.com/pytorch/pytorch/pull/32979)).
* [caffe2][quantization] Add initializer and precision as read-only property to QueryTensorQparam (#34706) ([#34706](https://github.com/pytorch/pytorch/pull/34706)).
* Fix missing header (#34762) ([#34762](https://github.com/pytorch/pytorch/pull/34762)).
* Add output_size argument to caffe2 Int8ResizeNearest (#30202) ([#30202](https://github.com/pytorch/pytorch/pull/30202)).
* Add support for converting quantized AvgPool2d and Reshape operations (#30490) ([#30490](https://github.com/pytorch/pytorch/pull/30490)).
* Add support for quantized slice conversion (#30498) ([#30498](https://github.com/pytorch/pytorch/pull/30498)).
* Call RandomNumberSeed() on-demand (#33539) ([#33539](https://github.com/pytorch/pytorch/pull/33539)).
* Add transfer_learning_blob_name_mappings into layer_model_helper to support layer model transfer learning ([#None](https://github.com/pytorch/pytorch/pull/None)).
* Revert D20289209: Support RowWiseSparseAdam on GPU ([#None](https://github.com/pytorch/pytorch/pull/None)).
