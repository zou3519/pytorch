* Skip undefined tensors when moving torch::nn module to a different device (#30523) ([#30523](https://github.com/pytorch/pytorch/pull/30523)).
* Exclude undefined tensors in the result of Module::parameters() / named_paramters() / buffers() / named_buffers() (#30626) ([#30626](https://github.com/pytorch/pytorch/pull/30626)).
* Remove namespace F = torch::nn::functional from torch/nn/modules/batchhnorm.h (#30684) ([#30684](https://github.com/pytorch/pytorch/pull/30684)).
* Add docs for how we expose declarations in at:: to torch:: (#30760) ([#30760](https://github.com/pytorch/pytorch/pull/30760)).
* added a serialize function in SGD class to utilize the existing macro for serialization/deserialization calls ([#30739](https://github.com/pytorch/pytorch/pull/30739)).
* Fix missing virtual destructor (#30927) ([#30927](https://github.com/pytorch/pytorch/pull/30927)).
* Added C++ API test (#30980) ([#30980](https://github.com/pytorch/pytorch/pull/30980)).
* Make Conv{1,2,3}dOptions and ConvTranspose{1,2,3}dOptions different classes (#31005) ([#31005](https://github.com/pytorch/pytorch/pull/31005)).
* C++ API parity: MultiheadAttention ([#27309](https://github.com/pytorch/pytorch/pull/27309)).
* C++ added 3rd arg of false to BatchNorm/InstanceNorm register_parameter … (#31873) ([#31873](https://github.com/pytorch/pytorch/pull/31873)).
* For torch::from_blob() add clue when memory is non-owned. (#31222) ([#31222](https://github.com/pytorch/pytorch/pull/31222)).
* C++ API parity: isinf (#31099) ([#31099](https://github.com/pytorch/pytorch/pull/31099)).
* C++ tensor indexing: add Slice / TensorIndex (#30424) ([#30424](https://github.com/pytorch/pytorch/pull/30424)).
* Add comments to torch::nn::ConvTranspose{1,2,3}d modules explaining how to use them in a Sequential module (#32223) ([#32223](https://github.com/pytorch/pytorch/pull/32223)).
* Adagrad optimizer - updated step function, added param_groups, state to optimizers ([#29335](https://github.com/pytorch/pytorch/pull/29335)).
* Bug fixes: torch::tensor(floating-point values) -> default dtype, and torch::tensor(integer values) ->at::kLong (#32367) ([#32367](https://github.com/pytorch/pytorch/pull/32367)).
* Fix torch::allclose to handle std::numeric_limits<T>::lowest() for integral types (#32978) ([#32978](https://github.com/pytorch/pytorch/pull/32978)).
* SGD: updated step and class design (#32592) ([#32592](https://github.com/pytorch/pytorch/pull/32592)).
* Fractional Max Pooling: output ratios defined as double (#33304) ([#33304](https://github.com/pytorch/pytorch/pull/33304)).
* Add at::Tensor::retain_grad API (#33349) ([#33349](https://github.com/pytorch/pytorch/pull/33349)).
* [C++ API] Expose AnyValue and AnyModuleHolder classes (#33026) ([#33026](https://github.com/pytorch/pytorch/pull/33026)).
* [C++ API] Allow skipping default arguments in module's forward method when module is used in Sequential (#33027) ([#33027](https://github.com/pytorch/pytorch/pull/33027)).
* Normalize reward-to-go in C++ actor-critic (#33550) ([#33550](https://github.com/pytorch/pytorch/pull/33550)).
* C++/Python API Parity: add pad_sequence (#32387) ([#32387](https://github.com/pytorch/pytorch/pull/32387)).
* Fix visibility of torch::nn::RNNImpl::options (#33718) ([#33718](https://github.com/pytorch/pytorch/pull/33718)).
* C++ tensor multi-dim indexing: add index() and index_put_() overloads, simple indexing tests, merge with Python indexing path (#32841) ([#32841](https://github.com/pytorch/pytorch/pull/32841)).
* [C++ API] Add PackedSequence / pack_padded_sequence / pad_packed_sequence / pack_sequence (#33652) ([#33652](https://github.com/pytorch/pytorch/pull/33652)).
* Add assert_tensor_equal and assert_tensor_not_equal to test/cpp/api/support.h (#30426) ([#30426](https://github.com/pytorch/pytorch/pull/30426)).
* C++ tensor indexing: more indexing tests (#30427) ([#30427](https://github.com/pytorch/pytorch/pull/30427)).
* [torch] Fix sign-compare warning in `torch::utils::rnn:pack_sequence` (#34185) ([#34185](https://github.com/pytorch/pytorch/pull/34185)).
* [C++ API Parity] Adam: updated step and class design (#33730) ([#33730](https://github.com/pytorch/pytorch/pull/33730)).
* C++ make torch::nn::Sequential push_back(AnyModule) methods public (#34208) ([#34208](https://github.com/pytorch/pytorch/pull/34208)).
* [C++ API] Remove init-list form of at::indexing::Slice (#34255) ([#34255](https://github.com/pytorch/pytorch/pull/34255)).
* [C++ API] Fix ModuleList compile error: error: 'begin' was not declared in this scope (#34463) ([#34463](https://github.com/pytorch/pytorch/pull/34463)).
* Remove `using namespace torch::autograd` from header files (#34423) ([#34423](https://github.com/pytorch/pytorch/pull/34423)).
* [C++ API Parity] rmsprop optimizer update (#33450) ([#33450](https://github.com/pytorch/pytorch/pull/33450)).
* [C++ API] Update torch::nn layer docs (#34522) ([#34522](https://github.com/pytorch/pytorch/pull/34522)).
* [C++ API] Update torch::nn functional docs (#34688) ([#34688](https://github.com/pytorch/pytorch/pull/34688)).
* [C++ API] Link to module options doc for functional that has same options as module (#34752) ([#34752](https://github.com/pytorch/pytorch/pull/34752)).
* [C++ API] RNNCell / LSTMCell / GRUCell layers (#34400) ([#34400](https://github.com/pytorch/pytorch/pull/34400)).
* [C++ API] RNN / GRU / LSTM layer refactoring (#34322) ([#34322](https://github.com/pytorch/pytorch/pull/34322)).
* [C++ API] RNN / GRU / LSTM layer refactoring (#34322) ([#34322](https://github.com/pytorch/pytorch/pull/34322)).
* [C++ API Parity] [Optimizers] added closure to optimizers (#34790) ([#34790](https://github.com/pytorch/pytorch/pull/34790)).
* [C++ API Parity] LBFGS optimizer step() update and added closure to the Optimizer step() function (#34564) ([#34564](https://github.com/pytorch/pytorch/pull/34564)).
* Split libtorch.so back into libtorch_{cpu,cuda,hip} (#30315) ([#30315](https://github.com/pytorch/pytorch/pull/30315)).
* Add missing _API definitions. (#30310) ([#30310](https://github.com/pytorch/pytorch/pull/30310)).
* Docs: c++11 -> c++14 (#30530) ([#30530](https://github.com/pytorch/pytorch/pull/30530)).
* [C++ API] AdaptiveLogSoftmaxWithLoss (#29076) ([#29076](https://github.com/pytorch/pytorch/pull/29076)).
* [C++ API] Remove deprecated torch::nn::BatchNorm / FeatureDropout / modules_ordered_dict and torch::nn::init::Nonlinearity / FanMode (#34508) ([#34508](https://github.com/pytorch/pytorch/pull/34508)).
* Ensure torch_cuda is linked against on Windows (#34288) ([#34288](https://github.com/pytorch/pytorch/pull/34288)).
* Update docs for cpp_extension on Windows (#30392) ([#30392](https://github.com/pytorch/pytorch/pull/30392)).
* TestCppExtension now removes /tmp/torch_extensions folder so that it can be used by other users in a multi-user environment. (#30095) ([#30095](https://github.com/pytorch/pytorch/pull/30095)).
* Fix race condition when creating build dir (#30956) ([#30956](https://github.com/pytorch/pytorch/pull/30956)).
* Uniformly apply Windows logic in cpp_extensions everywhere (#31161) ([#31161](https://github.com/pytorch/pytorch/pull/31161)).
* Make cuda search process of cpp extension quiet (#32620) ([#32620](https://github.com/pytorch/pytorch/pull/32620)).
* Remove -Werror from test/cpp_extensions/setup.py (#32704) ([#32704](https://github.com/pytorch/pytorch/pull/32704)).
* Build ahead-of-time C++ extensions with ninja on windows ([#33084](https://github.com/pytorch/pytorch/pull/33084)).
* Disable flaky test TestCppExtensionAOT.test_cuda_extension in Windows CI (#33282) ([#33282](https://github.com/pytorch/pytorch/pull/33282)).
* Add mechanism to pass a number of workers to cpp extensions (#33346) ([#33346](https://github.com/pytorch/pytorch/pull/33346)).
* Revert "Disable flaky test TestCppExtensionAOT.test_cuda_extension in… (#33404) ([#33404](https://github.com/pytorch/pytorch/pull/33404)).
* Fix overlapping keywords (#34142) ([#34142](https://github.com/pytorch/pytorch/pull/34142)).
* Fix C++ at::Tensor docs generation (#34467) ([#34467](https://github.com/pytorch/pytorch/pull/34467)).
* Update compiler warning about ABI compatibility (#34472) ([#34472](https://github.com/pytorch/pytorch/pull/34472)).
* Revert D20518647: [pytorch][PR] [C++ API Parity] [Optimizers] Merged Optimizer and LossClosureOptimizer ([#None](https://github.com/pytorch/pytorch/pull/None)).
* Revert D20524479: [pytorch][PR] [C++ API Parity] Add xor_convergence test for lbfgs ([#None](https://github.com/pytorch/pytorch/pull/None)).
* [C++ API Parity] [Optimizers] Merged Optimizer and LossClosureOptimizer (#34957) ([#34957](https://github.com/pytorch/pytorch/pull/34957)).
* [C++ API Parity] Add xor_convergence test for lbfgs (#35001) ([#35001](https://github.com/pytorch/pytorch/pull/35001)).
