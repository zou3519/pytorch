* refactor the observer removal and quantize tensor ([#30360](https://github.com/pytorch/pytorch/pull/30360)).
* Graph-mode quantization for convolution from traced model (#30245) ([#30245](https://github.com/pytorch/pytorch/pull/30245)).
* Updates to quantization documentation (#30288) ([#30288](https://github.com/pytorch/pytorch/pull/30288)).
* Fix quantized ConvReLU3d test (#30266) ([#30266](https://github.com/pytorch/pytorch/pull/30266)).
* Support to add dequant for each use of Value (#30145) ([#30145](https://github.com/pytorch/pytorch/pull/30145)).
* Add `get_qparams` and revert the changes to `calculate_qparams` (#30262) ([#30262](https://github.com/pytorch/pytorch/pull/30262)).
* Bug fix: Handle missing keys in observer state dict during load (#30357) ([#30357](https://github.com/pytorch/pytorch/pull/30357)).
* Fix BC for quantized linear ([#30481](https://github.com/pytorch/pytorch/pull/30481)).
* Fix docs so that the example works (#30120) ([#30120](https://github.com/pytorch/pytorch/pull/30120)).
* Add the explicit per-tensor/per-channel quant info when we print the module (#30591) ([#30591](https://github.com/pytorch/pytorch/pull/30591)).
* split getInvokedMethods (#30546) ([#30546](https://github.com/pytorch/pytorch/pull/30546)).
* DEFINE_DISPATCH in the correct namespace. (#30308) ([#30308](https://github.com/pytorch/pytorch/pull/30308)).
* Fix mapping white list (#30636) ([#30636](https://github.com/pytorch/pytorch/pull/30636)).
* Globally record observer nodes (#30547) ([#30547](https://github.com/pytorch/pytorch/pull/30547)).
* InsertObservers for shared class types (#30548) ([#30548](https://github.com/pytorch/pytorch/pull/30548)).
* Rename QuantizeHelper to InsertQuantDeQuantHelper (#30549) ([#30549](https://github.com/pytorch/pytorch/pull/30549)).
* Invoke more passes in `insertObservers` (#30473) ([#30473](https://github.com/pytorch/pytorch/pull/30473)).
* Supporting making submodules unique (#30037) ([#30037](https://github.com/pytorch/pytorch/pull/30037)).
* Make `InsertQuantDeQuantHelper` global (#30550) ([#30550](https://github.com/pytorch/pytorch/pull/30550)).
* Refactor bias and weight check and add aten::linear pattern (#30474) ([#30474](https://github.com/pytorch/pytorch/pull/30474)).
* Refactor test_quantization.py and enable `test_nested` (#30475) ([#30475](https://github.com/pytorch/pytorch/pull/30475)).
* Insert GetAttr for quantization parameters instead of Constant (#30551) ([#30551](https://github.com/pytorch/pytorch/pull/30551)).
* Add tests for quantizing traced models (#30476) ([#30476](https://github.com/pytorch/pytorch/pull/30476)).
* Add registerQParams function (#30552) ([#30552](https://github.com/pytorch/pytorch/pull/30552)).
* Remove `values_to_quantize_` (#30858) ([#30858](https://github.com/pytorch/pytorch/pull/30858)).
* Kill hypothesis deadline testing (#30890) ([#30890](https://github.com/pytorch/pytorch/pull/30890)).
* getQParams return a dictionary of qparams (#30859) ([#30859](https://github.com/pytorch/pytorch/pull/30859)).
* Factor out getInvokedMethod in `InsertQuantDeQuantHelper` (#30860) ([#30860](https://github.com/pytorch/pytorch/pull/30860)).
* qnnpack TanH ([#31013](https://github.com/pytorch/pytorch/pull/31013)).
* Remove `insert_prepack_unpack` and `fold_prepack` for now (#30909) ([#30909](https://github.com/pytorch/pytorch/pull/30909)).
* Adding quantized clamp kernel (#30541) ([#30541](https://github.com/pytorch/pytorch/pull/30541)).
* Remove redundant queries of qconfig in `insertObservers` (#31292) ([#31292](https://github.com/pytorch/pytorch/pull/31292)).
* Fix default instantation of dynamic quantized LSTM ([#31433](https://github.com/pytorch/pytorch/pull/31433)).
* Guard against copying from quantized Tensor to non-quantized Tensor (#29660) ([#29660](https://github.com/pytorch/pytorch/pull/29660)).
* Get QScheme from observer module (#31293) ([#31293](https://github.com/pytorch/pytorch/pull/31293)).
* Call `getQSchemeAndQParamMap` later in `quantizeTensors` (#31406) ([#31406](https://github.com/pytorch/pytorch/pull/31406)).
* Remove observers in the end (#31407) ([#31407](https://github.com/pytorch/pytorch/pull/31407)).
* Error out if legacy Tensor.new is called on alternate layouts / dtypes (#31485) ([#31485](https://github.com/pytorch/pytorch/pull/31485)).
* Fixed concatenation benchmark + added it to the microbenchmarking runs ([#31587](https://github.com/pytorch/pytorch/pull/31587)).
* Using _floats_wrapper in per_channel_tensor generation (#31780) ([#31780](https://github.com/pytorch/pytorch/pull/31780)).
* Enable foldbn tests (#29220) ([#29220](https://github.com/pytorch/pytorch/pull/29220)).
* Quantized H Tangent function (#31031) ([#31031](https://github.com/pytorch/pytorch/pull/31031)).
* Fix segfault in caffe2 slice test (#31801) ([#31801](https://github.com/pytorch/pytorch/pull/31801)).
* Quantized H Tangent function (#31031) ([#31031](https://github.com/pytorch/pytorch/pull/31031)).
* Remove qconfig_dict in top level eager mode quantization API (#31972) ([#31972](https://github.com/pytorch/pytorch/pull/31972)).
* Use default scale/zero_point in fake_quantize module instead of None (#32318) ([#32318](https://github.com/pytorch/pytorch/pull/32318)).
* Fix ASAN / potential segfault in quantized Tensor memory allocations. ([#29882](https://github.com/pytorch/pytorch/pull/29882)).
* Removed unused weight update in prepack. Moved zero point update to (#32254) ([#32254](https://github.com/pytorch/pytorch/pull/32254)).
* QNNPACK: Add support for dynamic quantization. ([#31896](https://github.com/pytorch/pytorch/pull/31896)).
* `insert_quant_dequant` pass support shared class types (#31408) ([#31408](https://github.com/pytorch/pytorch/pull/31408)).
* Move some of the helper functions for public use (#32202) ([#32202](https://github.com/pytorch/pytorch/pull/32202)).
* Adding QConfigTypePtrMap (#32203) ([#32203](https://github.com/pytorch/pytorch/pull/32203)).
* [quant][graphmode] Default to non-inplace in graph mode quantization API (#32204) ([#32204](https://github.com/pytorch/pytorch/pull/32204)).
* [quant][graphmode] Support quantizing shared ClassType with different qconfigs (#32205) ([#32205](https://github.com/pytorch/pytorch/pull/32205)).
* [quant] Re-enable test_nested that has different qconfig for shared ClassType (#32206) ([#32206](https://github.com/pytorch/pytorch/pull/32206)).
* [quant] Re-enable fold_convbn in quantize_script (#32302) ([#32302](https://github.com/pytorch/pytorch/pull/32302)).
* [quant][graphmode] Call _jit_pass_dedup_module_ueses in quantize_script (#32303) ([#32303](https://github.com/pytorch/pytorch/pull/32303)).
* [refactor] Adding FoldConvBatchNorm2dHelper (#32374) ([#32374](https://github.com/pytorch/pytorch/pull/32374)).
* [quant][graphmode][refactor] Better API for fold_convbn (#32380) ([#32380](https://github.com/pytorch/pytorch/pull/32380)).
* Add operator support for dynamic quant on mobile (#32479) ([#32479](https://github.com/pytorch/pytorch/pull/32479)).
* Adding native qconcat ([#32252](https://github.com/pytorch/pytorch/pull/32252)).
* [quantization] FP16 dynamic quantized Linear ([#32331](https://github.com/pytorch/pytorch/pull/32331)).
* Temporarily disable the test_quantized_rnn test (#32742) ([#32742](https://github.com/pytorch/pytorch/pull/32742)).
* [quantization] Remove incorrect fp16 dynamic linear/relu op ([#32774](https://github.com/pytorch/pytorch/pull/32774)).
* Add support for Dynamic LSTM quantization on Mobile (#32757) ([#32757](https://github.com/pytorch/pytorch/pull/32757)).
* Resubmit more code fakefp16 mapping unification (#32798) ([#32798](https://github.com/pytorch/pytorch/pull/32798)).
* Don't serialize None values in observer (#32733) ([#32733](https://github.com/pytorch/pytorch/pull/32733)).
* Quantized sigmoid function ([#31851](https://github.com/pytorch/pytorch/pull/31851)).
* TorchScript add check if quantized ([#32890](https://github.com/pytorch/pytorch/pull/32890)).
* Quantized leaky relu ([#33004](https://github.com/pytorch/pytorch/pull/33004)).
* [pytorch][quant] Add assert for min, max, qmin, qmax for ChooseQuantizationParams (#32739) ([#32739](https://github.com/pytorch/pytorch/pull/32739)).
* Add histogram collection and weight prepacking utils (#33125) ([#33125](https://github.com/pytorch/pytorch/pull/33125)).
* [quant] Add a quantized batch_norm operator (#33080) ([#33080](https://github.com/pytorch/pytorch/pull/33080)).
* [quant] Add Quantized BatchNorm2d module (#33109) ([#33109](https://github.com/pytorch/pytorch/pull/33109)).
* Enable inplace relu fusion for training (#33105) ([#33105](https://github.com/pytorch/pytorch/pull/33105)).
* Bug fix in dynamic quantization kernels + better test coverage. (#33320) ([#33320](https://github.com/pytorch/pytorch/pull/33320)).
* [pt][fbgemm] Turn on USE_FBGEMM on Windows env (#297) ([#33250](https://github.com/pytorch/pytorch/pull/33250)).
* avoid large vector copy when query per_channel q_params (#31040) ([#31040](https://github.com/pytorch/pytorch/pull/31040)).
* Support broadcast for quantized mul kernel (#30442) ([#30442](https://github.com/pytorch/pytorch/pull/30442)).
* [quant] Regsiter fake_quant and observer attributes as buffers (#33626) ([#33626](https://github.com/pytorch/pytorch/pull/33626)).
* [pt][quant] RNN debug test (#33621) ([#33621](https://github.com/pytorch/pytorch/pull/33621)).
* [quant][graphmode] FoldConvBatchNorm2d support shared ClassTypes (#32379) ([#32379](https://github.com/pytorch/pytorch/pull/32379)).
* [quant] Make FakeQuant use REGISTER_DISPATCH (#33682) ([#33682](https://github.com/pytorch/pytorch/pull/33682)).
* [quant][graphmode][refactor] Move the check for qconfig inside insertObserver call (#32809) ([#32809](https://github.com/pytorch/pytorch/pull/32809)).
* [pytorch] Set alias analysis kind to FROM_SCHEMA for qadd, qmul, qclamp, qconcat (#33359) ([#33359](https://github.com/pytorch/pytorch/pull/33359)).
* [quant][graphmode][refactor] Change signature of getModuleAccessPath (#32812) ([#32812](https://github.com/pytorch/pytorch/pull/32812)).
* Back out "[pt][quant] RNN debug test" (#33750) ([#33750](https://github.com/pytorch/pytorch/pull/33750)).
* Migrate fake_quant_slice to TensorIterator (#33744) ([#33744](https://github.com/pytorch/pytorch/pull/33744)).
* [quant][graphmode][refactor] Separate preprocess step for insertObserver (#32813) ([#32813](https://github.com/pytorch/pytorch/pull/32813)).
* [quant][graphmode] refactor nodeQuantizable (#33171) ([#33171](https://github.com/pytorch/pytorch/pull/33171)).
* [quant][graphmode][refactor] Factor out insertDequantCall (#33172) ([#33172](https://github.com/pytorch/pytorch/pull/33172)).
* Disable printing of the histogram when dump (#33749) ([#33749](https://github.com/pytorch/pytorch/pull/33749)).
* [pt][quant] Parallelize quantize and dequantize (#33765) ([#33765](https://github.com/pytorch/pytorch/pull/33765)).
* Per channel quantization performance improvement (#33772) ([#33772](https://github.com/pytorch/pytorch/pull/33772)).
* Preserve Backward compatibility of models serialized before #31040 (#33796) ([#33796](https://github.com/pytorch/pytorch/pull/33796)).
* [quant][graphmode] Replicate dequantize nodes (#33531) ([#33531](https://github.com/pytorch/pytorch/pull/33531)).
* [quant][graphmode] Swap dequantize after inline for ops that doesn't require observation (#33173) ([#33173](https://github.com/pytorch/pytorch/pull/33173)).
* [quant][graphmode][refactor] Simplify signature for insertObserverFor (#33274) ([#33274](https://github.com/pytorch/pytorch/pull/33274)).
* [quant][graphmode][refactor] Move values_to_skip check inside valueNeedsToBeQuantized (#33275) ([#33275](https://github.com/pytorch/pytorch/pull/33275)).
* [quant][graphmode][refactor] Checks for bias and weight (#33273) ([#33273](https://github.com/pytorch/pytorch/pull/33273)).
* [quant][graphmode][refactor] Move check for weight outside of insertObserverFor (#33276) ([#33276](https://github.com/pytorch/pytorch/pull/33276)).
* [quant][graphmode][refactor] Factor out getInvokedMethod (#33649) ([#33649](https://github.com/pytorch/pytorch/pull/33649)).
* [quant][graphmode] Observing input/output values in call site (#33277) ([#33277](https://github.com/pytorch/pytorch/pull/33277)).
* Fix HistogramObserver to not do detach on input (#34114) ([#34114](https://github.com/pytorch/pytorch/pull/34114)).
* [quant][graphmode] Skip quantizing input and output in matched module (#32814) ([#32814](https://github.com/pytorch/pytorch/pull/32814)).
* [quant][graphmode] Add add_relu pattern in skip values (#32816) ([#32816](https://github.com/pytorch/pytorch/pull/32816)).
* [quant] Run weight_post_process for QAT (#33852) ([#33852](https://github.com/pytorch/pytorch/pull/33852)).
* Tuck the packing logic into Int8FCPackWeight op (#34289) ([#34289](https://github.com/pytorch/pytorch/pull/34289)).
* [quant] Fix histogram observer to work with QAT on GPU (#34232) ([#34232](https://github.com/pytorch/pytorch/pull/34232)).
* [quant] Speed up per-channel min-max observer (#34118) ([#34118](https://github.com/pytorch/pytorch/pull/34118)).
* [PyTorch] Remove const modifiers from passed by value integers in qbatch_norm_fn (#34378) ([#34378](https://github.com/pytorch/pytorch/pull/34378)).
* Add the 3d avg pool for video related model (#33339) ([#33339](https://github.com/pytorch/pytorch/pull/33339)).
* Fixed typos in quantization docs / docstrings (#34182) ([#34182](https://github.com/pytorch/pytorch/pull/34182)).
* [quantization] Make FP16 RNN use new prepack op (#34339) ([#34339](https://github.com/pytorch/pytorch/pull/34339)).
* [quant][graphmode] Handling ops doesn't require observation in insertObservers (#33481) ([#33481](https://github.com/pytorch/pytorch/pull/33481)).
* [quant][graphmode] Swap quantized functional linear with aten::linear (#33853) ([#33853](https://github.com/pytorch/pytorch/pull/33853)).
* [pt][quant] Vectorized qmul and more methods on qint data types (#34376) ([#34376](https://github.com/pytorch/pytorch/pull/34376)).
* add quantized_hardtanh (#34097) ([#34097](https://github.com/pytorch/pytorch/pull/34097)).
* fix the quantized batchnorm2d (#34579) ([#34579](https://github.com/pytorch/pytorch/pull/34579)).
* add quantized ELU activation (#34267) ([#34267](https://github.com/pytorch/pytorch/pull/34267)).
* [quant][graphmode] Add Finalize function that inlines graph and produce quantized ops (#33927) ([#33927](https://github.com/pytorch/pytorch/pull/33927)).
* [quant][graphmode] Add quantized conv2d-relu fusion pattern (#33279) ([#33279](https://github.com/pytorch/pytorch/pull/33279)).
* Add the 3d upsample quantized op for video model (#34594) ([#34594](https://github.com/pytorch/pytorch/pull/34594)).
* Add the quantized batch_norm3d and also batch_norm3d fused with relu operators (#34702) ([#34702](https://github.com/pytorch/pytorch/pull/34702)).
* [reland][quant][graphmode] Add quantized conv2d-relu fusion pattern (#33279) (#34744) ([#34744](https://github.com/pytorch/pytorch/pull/34744)).
* JIT pass to insert XNNPACK ops (#34048) ([#34048](https://github.com/pytorch/pytorch/pull/34048)).
* [quant][graphmode] Add quantization pattern for quantized::add_relu (#33532) ([#33532](https://github.com/pytorch/pytorch/pull/33532)).
* [quant][mobile] Not use qnnpack max_pool2d if ceil_mode is true (#34844) ([#34844](https://github.com/pytorch/pytorch/pull/34844)).
* adds quantized implementation of hard sigmoid (#34607) ([#34607](https://github.com/pytorch/pytorch/pull/34607)).
* [quant][graphmode] Quantization pattern for aten::linear (#33854) ([#33854](https://github.com/pytorch/pytorch/pull/33854)).
* [quant][graphmode][refactor] Change QParamMap to QParamVector (#34314) ([#34314](https://github.com/pytorch/pytorch/pull/34314)).
* Refactor QAT Conv module for better extensibility (#30362) ([#30362](https://github.com/pytorch/pytorch/pull/30362)).
* Fix test case after get_qparams refactor (#30470) ([#30470](https://github.com/pytorch/pytorch/pull/30470)).
* Disable test_backward_per_tensor in test_fake_quant (#30594) ([#30594](https://github.com/pytorch/pytorch/pull/30594)).
* Temporarily disable test_numerical_consistency_per_tensor (#30600) ([#30600](https://github.com/pytorch/pytorch/pull/30600)).
* Raise an error for is_signed on quantized types (#30527) ([#30527](https://github.com/pytorch/pytorch/pull/30527)).
* Move QScheme ops to c10 (#30134) ([#30134](https://github.com/pytorch/pytorch/pull/30134)).
* dynamicly quantized linear benchmarking ([#30148](https://github.com/pytorch/pytorch/pull/30148)).
* dynamicly quantized lstm benchmarking ([#30149](https://github.com/pytorch/pytorch/pull/30149)).
* Docs entry for the `is_quantized` ([#32075](https://github.com/pytorch/pytorch/pull/32075)).
* [fix] use non-inplace for insert observer pass (#34190) ([#34190](https://github.com/pytorch/pytorch/pull/34190)).
* [quant][graphmode] quantization support for aten::rehshape (#34803) ([#34803](https://github.com/pytorch/pytorch/pull/34803)).
* [quant][graphmode] Add quantization support for aten::dropout (#34347) ([#34347](https://github.com/pytorch/pytorch/pull/34347)).
* [quant][graphmode] insert quant/dequant work for duplicated debugName (#34315) ([#34315](https://github.com/pytorch/pytorch/pull/34315)).
