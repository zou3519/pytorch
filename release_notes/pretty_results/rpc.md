* Add RpcAgent::getWorkerInfos() (#30241) ([#30241](https://github.com/pytorch/pytorch/pull/30241)).
* add default arg for init_method (#30208) ([#30208](https://github.com/pytorch/pytorch/pull/30208)).
* By default ignore RRef leaks during shutdown (#30217) ([#30217](https://github.com/pytorch/pytorch/pull/30217)).
* Add local shutdown to process group agent (#30330) ([#30330](https://github.com/pytorch/pytorch/pull/30330)).
* Improve process_group_agent() serialization speed (#29785) ([#29785](https://github.com/pytorch/pytorch/pull/29785)).
* Reorganize rpc API doc and add introduction (#30491) ([#30491](https://github.com/pytorch/pytorch/pull/30491)).
* Make doc source format consistent in rpc/init.cpp ([#30515](https://github.com/pytorch/pytorch/pull/30515)).
* Add examples to RRef doc ([#30516](https://github.com/pytorch/pytorch/pull/30516)).
* Fix serialization memory lifetime issue. (#30603) ([#30603](https://github.com/pytorch/pytorch/pull/30603)).
* Remove deprecated fromIvalue in RRefForkData ([#30646](https://github.com/pytorch/pytorch/pull/30646)).
* Change test_invalid_names test to only test constructor of WorkerInfo (#30620) ([#30620](https://github.com/pytorch/pytorch/pull/30620)).
* remove default constructor in futureInfo (#30197) ([#30197](https://github.com/pytorch/pytorch/pull/30197)).
* Add RRef.__str__() API ([#30609](https://github.com/pytorch/pytorch/pull/30609)).
* Adding Debug Info for RRef Context ([#30610](https://github.com/pytorch/pytorch/pull/30610)).
* modify test_local_shutdown_with_rpc to not be flaky (#30837) ([#30837](https://github.com/pytorch/pytorch/pull/30837)).
* Add get_metrics and get_debug_info to rpc agent (#30833) ([#30833](https://github.com/pytorch/pytorch/pull/30833)).
* Replace deprecated AT_ERROR with TORCH_CHECK to reduce warnings in rpc ([#30794](https://github.com/pytorch/pytorch/pull/30794)).
* Add more details to explain rpc_backend_options arg in init_rpc ([#30855](https://github.com/pytorch/pytorch/pull/30855)).
* Fix examples in API doc ([#30856](https://github.com/pytorch/pytorch/pull/30856)).
* Fix examples in RRef API doc ([#30857](https://github.com/pytorch/pytorch/pull/30857)).
* Adding debugging metrics to process group agent ([#30884](https://github.com/pytorch/pytorch/pull/30884)).
* Add glue code to collect debug info from all components ([#30888](https://github.com/pytorch/pytorch/pull/30888)).
* Disable flaky test_rref_context_debug_info ([#30990](https://github.com/pytorch/pytorch/pull/30990)).
* Allow async work in rpc RequestCallback processing. (#30637) ([#30637](https://github.com/pytorch/pytorch/pull/30637)).
* Document WorkerInfo and RpcBackendOptions structures in RPC docs. (#31077) ([#31077](https://github.com/pytorch/pytorch/pull/31077)).
* Re-enable test_rref_context_debug_info after enforcing proper synchronization (#30994) ([#30994](https://github.com/pytorch/pytorch/pull/30994)).
* Change startTime_ to endTime_ in FutureInfo (#30342) ([#30342](https://github.com/pytorch/pytorch/pull/30342)).
* Make rref fetch calls async. (#31086) ([#31086](https://github.com/pytorch/pytorch/pull/31086)).
* Change type of timeoutFutures_ key to time_point instead of duration (#31078) ([#31078](https://github.com/pytorch/pytorch/pull/31078)).
* Tweak pollTimedOutRPCs thread synchronization (#30355) ([#30355](https://github.com/pytorch/pytorch/pull/30355)).
* Only create OwnerRRefs when processing remote calls (#31163) ([#31163](https://github.com/pytorch/pytorch/pull/31163)).
* build a generic future<T> (#29579) ([#29579](https://github.com/pytorch/pytorch/pull/29579)).
* Robustify rpc_agent handlers with generic Future<T> (#31224) ([#31224](https://github.com/pytorch/pytorch/pull/31224)).
* Fix error message in incorrect rref.localValue() call (#31199) ([#31199](https://github.com/pytorch/pytorch/pull/31199)).
* Use global lock instead of per instance lock. (#31404) ([#31404](https://github.com/pytorch/pytorch/pull/31404)).
* Avoid sending large unneeded data over wire in process_group_agent. (#31357) ([#31357](https://github.com/pytorch/pytorch/pull/31357)).
* Kill MessageType::SHUTDOWN related logic in pg agent (#31270) ([#31270](https://github.com/pytorch/pytorch/pull/31270)).
* fix test_process_group_debug_info flaky test (#31533) ([#31533](https://github.com/pytorch/pytorch/pull/31533)).
* Split RRef class into abstract RRef and RRefBase (#28942) ([#28942](https://github.com/pytorch/pytorch/pull/28942)).
* Fix flaky test_debug_info. (#31675) ([#31675](https://github.com/pytorch/pytorch/pull/31675)).
* Apply clang-format to csrc/distributed/rpc ([#31681](https://github.com/pytorch/pytorch/pull/31681)).
* add num_pending_users to debug info (#31539) ([#31539](https://github.com/pytorch/pytorch/pull/31539)).
* use unordered_set instead of vector for futureTimeouts key in (#31813) ([#31813](https://github.com/pytorch/pytorch/pull/31813)).
* Disable flaky test_debug_info ([#31847](https://github.com/pytorch/pytorch/pull/31847)).
* Implement backend-agnostic rpc._wait_all_workers() utility (#30710) ([#30710](https://github.com/pytorch/pytorch/pull/30710)).
* Integrate async mode for autograd engine with distributed autograd. (#31508) ([#31508](https://github.com/pytorch/pytorch/pull/31508)).
* Make RRef leak detection always print a warning log (#31922) ([#31922](https://github.com/pytorch/pytorch/pull/31922)).
* Implement backend-agnostic rpc._wait_all_workers() utility (#31888) ([#31888](https://github.com/pytorch/pytorch/pull/31888)).
* Explain RPC behavior when using Tensor as arg or return value ([#31968](https://github.com/pytorch/pytorch/pull/31968)).
* add missing braces for format in rpc _to_worker_info (#31969) ([#31969](https://github.com/pytorch/pytorch/pull/31969)).
* catch exceptions in ProcessGroupAgent::enqueueSend and report them. (#31023) ([#31023](https://github.com/pytorch/pytorch/pull/31023)).
* enable autograd profiler to work with RPC and RRef. (#31381) ([#31381](https://github.com/pytorch/pytorch/pull/31381)).
* support torch script call over rpc (#30063) ([#30063](https://github.com/pytorch/pytorch/pull/30063)).
* Implement backend-agnostic rpc._wait_all_workers() utility (#32190) ([#32190](https://github.com/pytorch/pytorch/pull/32190)).
* support torch script call over rpc (#32197) ([#32197](https://github.com/pytorch/pytorch/pull/32197)).
* add an option to record time spent waiting for GIL (#30842) ([#30842](https://github.com/pytorch/pytorch/pull/30842)).
* Move pytorch distributed tests to separate folder for contbuild. (#30445) ([#30445](https://github.com/pytorch/pytorch/pull/30445)).
* [rpc] Remove template on RRef and add Type to RRef creation (#30630) ([#30630](https://github.com/pytorch/pytorch/pull/30630)).
* [pytorch] Minor: boilerplate to propagate errors in request_callback_impl (#32556) ([#32556](https://github.com/pytorch/pytorch/pull/32556)).
* Support TorchScript call over remote API (RRef) (#32466) ([#32466](https://github.com/pytorch/pytorch/pull/32466)).
* [rpc] make handling of FORWARD_AUTOGRAD_REQ in request_callback_impl (#32476) ([#32476](https://github.com/pytorch/pytorch/pull/32476)).
* Make _wait_all_workers() support being called for multiple times (#32624) ([#32624](https://github.com/pytorch/pytorch/pull/32624)).
* Make C++ RpcAgent::currentRPCAgent_ the source of truth of current RPC Agent (#32633) ([#32633](https://github.com/pytorch/pytorch/pull/32633)).
* Use the C++ current RpcAgent pointer to eliminate the unnecessary argument passing from Python world (#32635) ([#32635](https://github.com/pytorch/pytorch/pull/32635)).
* [rpc][easy] remove redundant test in rpc_test.py (#32588) ([#32588](https://github.com/pytorch/pytorch/pull/32588)).
* [rpc][flaky-tests] fix for test_handle_send_exceptions and (#32656) ([#32656](https://github.com/pytorch/pytorch/pull/32656)).
* apply linter to rpc test files (#32659) ([#32659](https://github.com/pytorch/pytorch/pull/32659)).
* clean up GIL usuage (#32748) ([#32748](https://github.com/pytorch/pytorch/pull/32748)).
* [pytorch] Minor: avoid acquiring GIL twice in PyRRef::localValue() (#32785) ([#32785](https://github.com/pytorch/pytorch/pull/32785)).
* Reduce RPC branches for Python/BuiltinOp/TorchScript (#32689) ([#32689](https://github.com/pytorch/pytorch/pull/32689)).
* Remove GIL from RRefContext (#32807) ([#32807](https://github.com/pytorch/pytorch/pull/32807)).
* Remove Python dependency (toPyTuple/fromPyTuple, jitCompilationUnit, deserialize) in rref_impl.h/cpp (#32753) ([#32753](https://github.com/pytorch/pytorch/pull/32753)).
* Fixed the flaky test_rref_context_debug_info (#32749) ([#32749](https://github.com/pytorch/pytorch/pull/32749)).
* [rpc] don't crash callee when function does not exist on it, instead return Exception (#32726) ([#32726](https://github.com/pytorch/pytorch/pull/32726)).
* Use leaky singletons for torch.distributed. (#32923) ([#32923](https://github.com/pytorch/pytorch/pull/32923)).
* [rpc] throw correct Exception on local client based on the RemoteException (#32936) ([#32936](https://github.com/pytorch/pytorch/pull/32936)).
* Distributed Autograd: Allow multiple backward passes to accumulate gradients. (#32506) ([#32506](https://github.com/pytorch/pytorch/pull/32506)).
* [rpc][easy] move unnecessary python call directly to pybind (#33174) ([#33174](https://github.com/pytorch/pytorch/pull/33174)).
* handle errors in ProcessGroupAgent::listenLoop(). (#32957) ([#32957](https://github.com/pytorch/pytorch/pull/32957)).
* [rpc] Switch RRef to be managed by intrusive_ptr (#33189) ([#33189](https://github.com/pytorch/pytorch/pull/33189)).
* [jit] Initial use RRef in TorchScript (#33190) ([#33190](https://github.com/pytorch/pytorch/pull/33190)).
* [RPC Reliability] Implemented retries for RPCs with exponential backoff (#32602) ([#32602](https://github.com/pytorch/pytorch/pull/32602)).
* Make RPC message constructor actually move (#33440) ([#33440](https://github.com/pytorch/pytorch/pull/33440)).
* [RPC Reliability] Enabled retries for RPCs with exponential backoff (#33365) ([#33365](https://github.com/pytorch/pytorch/pull/33365)).
* allow remote torchscript call to itself (#32990) ([#32990](https://github.com/pytorch/pytorch/pull/32990)).
* [pytorch] Minor: add GIL assert to PythonRpcHandler::handleExceptionGILHeld (#33557) ([#33557](https://github.com/pytorch/pytorch/pull/33557)).
* jit pickling rref (#32959) ([#32959](https://github.com/pytorch/pytorch/pull/32959)).
* Add missing test launchers for JitRpcTest and JitDistAutogradTest (#32891) ([#32891](https://github.com/pytorch/pytorch/pull/32891)).
* [jit] infer RRef type as container type (#33369) ([#33369](https://github.com/pytorch/pytorch/pull/33369)).
* [jit] make RRef type annotation available in Python (#33526) ([#33526](https://github.com/pytorch/pytorch/pull/33526)).
* [rpc] special case tensor type check when getting RRef (#33582) ([#33582](https://github.com/pytorch/pytorch/pull/33582)).
* [Dist Autograd] Functional API for Dist Autograd and Dist Optimizer (#33711) ([#33711](https://github.com/pytorch/pytorch/pull/33711)).
* Release GIL for RPC pybind functions. (#33610) ([#33610](https://github.com/pytorch/pytorch/pull/33610)).
* [Revert] manual revert of D19918320 (#33920) ([#33920](https://github.com/pytorch/pytorch/pull/33920)).
* [jit] allow RRef local creation with IValue objects (#33263) ([#33263](https://github.com/pytorch/pytorch/pull/33263)).
* [resubmit] try to infer rref type from python (#33992) ([#33992](https://github.com/pytorch/pytorch/pull/33992)).
* move test helper functions out of test funciton (#33960) ([#33960](https://github.com/pytorch/pytorch/pull/33960)).
* [rpc][metrics] add initial metric handler classes. (#33153) ([#33153](https://github.com/pytorch/pytorch/pull/33153)).
* Improve ProcessGroup RpcBackendOptions Constructor API (#34081) ([#34081](https://github.com/pytorch/pytorch/pull/34081)).
* Use double quotes in C++ to stay consistent with Python RPC docs (#34095) ([#34095](https://github.com/pytorch/pytorch/pull/34095)).
* Apply clang-format to RPC files (#34139) ([#34139](https://github.com/pytorch/pytorch/pull/34139)).
* [JIT] Register rpc.rpc_async(..) as a JIT operator (#33329) ([#33329](https://github.com/pytorch/pytorch/pull/33329)).
* [Dist Autograd][Better Engineering] Enhanced Error Reporting in Dist Autograd/RPC (#34179) ([#34179](https://github.com/pytorch/pytorch/pull/34179)).
* Add test to verify dist_autograd doesn't populate .grad field. (#33949) ([#33949](https://github.com/pytorch/pytorch/pull/33949)).
* Remove RPC TorchScript private API (#33978) ([#33978](https://github.com/pytorch/pytorch/pull/33978)).
* [RPC] Create local RRef<ModuleInterface> remotely in Python, use it remotely in TorchScript (#34183) ([#34183](https://github.com/pytorch/pytorch/pull/34183)).
* Add worker_name helper to dist_utils. (#34162) ([#34162](https://github.com/pytorch/pytorch/pull/34162)).
* [RpcAgent] Metrics for current num active/async rpc calls. (#34398) ([#34398](https://github.com/pytorch/pytorch/pull/34398)).
* Avoid copy contents in SerializedPyObj (#34490) ([#34490](https://github.com/pytorch/pytorch/pull/34490)).
* Consolidate Python Messages to use SerializedPyObj (#34491) ([#34491](https://github.com/pytorch/pytorch/pull/34491)).
* Remove _load_return_value from RPC internal.py (#34492) ([#34492](https://github.com/pytorch/pytorch/pull/34492)).
* Use SerializedPyObj in PythonRpcHandler (#34493) ([#34493](https://github.com/pytorch/pytorch/pull/34493)).
* Split deserialize from _run_function in RPC internal.py (#34494) ([#34494](https://github.com/pytorch/pytorch/pull/34494)).
* Use SerializedPyObj in PythonRpcHandler::generatePythonUDFResult (#34495) ([#34495](https://github.com/pytorch/pytorch/pull/34495)).
* Unify gradient accumulation between distributed autograd and local autograd (#33214) ([#33214](https://github.com/pytorch/pytorch/pull/33214)).
* Fix static data initialization deadlock on GIL (#34505) ([#34505](https://github.com/pytorch/pytorch/pull/34505)).
* Delete all user forks tracked in RRefContext before graceful shutting down (#31893) ([#31893](https://github.com/pytorch/pytorch/pull/31893)).
* [pytorch-rpc] WireSerializer should check has_storage() (#34626) ([#34626](https://github.com/pytorch/pytorch/pull/34626)).
* Use c10::str in process_group_agent.cpp (#34679) ([#34679](https://github.com/pytorch/pytorch/pull/34679)).
* Use c10::str in py_rref.cpp (#34681) ([#34681](https://github.com/pytorch/pytorch/pull/34681)).
* Best-effort Error Detection for Using Deleted UserRRefs (#34673) ([#34673](https://github.com/pytorch/pytorch/pull/34673)).
* [docs][1.5] update RPC docs to reflect correct use of dist_autograd backwards and dist_optim step() (#34670) ([#34670](https://github.com/pytorch/pytorch/pull/34670)).
* [rpc][profiler] add a test case to verify record_function context manager works (#34511) ([#34511](https://github.com/pytorch/pytorch/pull/34511)).
* Disallow sending CUDA tensors over RPC for current RPC agents. (#33604) ([#33604](https://github.com/pytorch/pytorch/pull/33604)).
* Split deserialize from runPythonUdf and remove generatePythonUDFResult (#34496) ([#34496](https://github.com/pytorch/pytorch/pull/34496)).
* Fix send count for local RPC (#34809) ([#34809](https://github.com/pytorch/pytorch/pull/34809)).
* Don't run user function until all UserRRefs in the args are confirmed (#34497) ([#34497](https://github.com/pytorch/pytorch/pull/34497)).
* fix tests that could have racy script module instantiation (#34792) ([#34792](https://github.com/pytorch/pytorch/pull/34792)).
* [RPC] Use qualified name str directly in RPC torch script code path (#34733) ([#34733](https://github.com/pytorch/pytorch/pull/34733)).
* [RPC] Avoid polluting Python root logger on importing "torch" module (#34871) ([#34871](https://github.com/pytorch/pytorch/pull/34871)).
* Add a warning for RRef serialization (#34884) ([#34884](https://github.com/pytorch/pytorch/pull/34884)).
* Adding warnings for async Tensor serialization in remote and rpc_async (#34885) ([#34885](https://github.com/pytorch/pytorch/pull/34885)).
* Removing experimental tag in for RPC and adding experimental tag for RPC+TorchScript (#34887) ([#34887](https://github.com/pytorch/pytorch/pull/34887)).
* Update descriptions for transmitting CUDA tensors (#34888) ([#34888](https://github.com/pytorch/pytorch/pull/34888)).
* Minor fixes for RPC API docs (#34890) ([#34890](https://github.com/pytorch/pytorch/pull/34890)).
* Fix example block format in Distributed Optimizer API doc (#34919) ([#34919](https://github.com/pytorch/pytorch/pull/34919)).
* Fix dist autograd context Example block format (#34921) ([#34921](https://github.com/pytorch/pytorch/pull/34921)).
* fix barrier in jit test (#34901) ([#34901](https://github.com/pytorch/pytorch/pull/34901)).
* [rpc] handle exceptions in ProcessGroupAgent::enqueueRecv (#34413) ([#34413](https://github.com/pytorch/pytorch/pull/34413)).
* [JIT] Make RPC RRef Owner WorkerInfo.name available to TorchScript (#34896) ([#34896](https://github.com/pytorch/pytorch/pull/34896)).
* Support using self as the destination in rpc.remote for builtin operators (#34931) ([#34931](https://github.com/pytorch/pytorch/pull/34931)).
* Don't build test_cpp_rpc if torch is built without distributed support (#30587) ([#30587](https://github.com/pytorch/pytorch/pull/30587)).
* Setup operator registration for distributed package (#31214) ([#31214](https://github.com/pytorch/pytorch/pull/31214)).
* Attach autograd edges only for tensors requiring grad. (#30904) ([#30904](https://github.com/pytorch/pytorch/pull/30904)).
* Add debug info API for distributed autograd. (#30642) ([#30642](https://github.com/pytorch/pytorch/pull/30642)).
* add the worker IDs outside of addSendRpcBackward to ensure they are (#30914) ([#30914](https://github.com/pytorch/pytorch/pull/30914)).
* Fix memory leak due to circular dependency. (#31030) ([#31030](https://github.com/pytorch/pytorch/pull/31030)).
* Remove unused argument "destId" in addSendRpcBackward (#31207) ([#31207](https://github.com/pytorch/pytorch/pull/31207)).
* Remove the second copy on calling dist_autograd_context._known_worker_ids() (#31206) ([#31206](https://github.com/pytorch/pytorch/pull/31206)).
* Provide async mode for local autograd engine. (#31230) ([#31230](https://github.com/pytorch/pytorch/pull/31230)).
* [Pytorch] Propagate errors in clearAndWaitForOutstandingRpcsAsync. (#32952) ([#32952](https://github.com/pytorch/pytorch/pull/32952)).
* Fix example format in Distributed Autograd doc (#34914) ([#34914](https://github.com/pytorch/pytorch/pull/34914)).
* Make RecordFunction more robust for async use cases (#34122) ([#34122](https://github.com/pytorch/pytorch/pull/34122)).
* Add barriers to fix flaky test_graph_for_py_nested_call and (#30624) ([#30624](https://github.com/pytorch/pytorch/pull/30624)).
* Enable test_trainer_ps in dist_autograd_test.py ([#30341](https://github.com/pytorch/pytorch/pull/30341)).
* Fix lint issues in dist_autograd_test.py (#30928) ([#30928](https://github.com/pytorch/pytorch/pull/30928)).
* CODEOWNERS for distributed optimizer. (#31403) ([#31403](https://github.com/pytorch/pytorch/pull/31403)).
* fix test_backward_node_failure flakiness (#31588) ([#31588](https://github.com/pytorch/pytorch/pull/31588)).
* Refactor tests in pytorch's test/dist_autograd_test.py file (#31803) ([#31803](https://github.com/pytorch/pytorch/pull/31803)).
* minor doc tweak to use mp.spawn in example (#30381) ([#30381](https://github.com/pytorch/pytorch/pull/30381)).
* Add missing `shuffle` attribute to DistributedSampler typing file ([#28763](https://github.com/pytorch/pytorch/pull/28763)).
* Add warning and example for seeding to DistributedSampler (#32951) ([#32951](https://github.com/pytorch/pytorch/pull/32951)).
* Fix backward compatibility tests (#34071) ([#34071](https://github.com/pytorch/pytorch/pull/34071)).
* Run RPC JIT tests with variable type hints only in Python >=3.6 (#34284) ([#34284](https://github.com/pytorch/pytorch/pull/34284)).
* [docs][1.5] Update distributed autograd note (#34657) ([#34657](https://github.com/pytorch/pytorch/pull/34657)).
* [profiler][rpc] fix a race condition in the profiler when multiple threads call (#33719) ([#33719](https://github.com/pytorch/pytorch/pull/33719)).
* Revert D20164420: [1.5 Release][Dist Autograd][Better Engineering] Notify Workers on Failure during Distributed Autograd ([#None](https://github.com/pytorch/pytorch/pull/None)).
* Revert D7778113: Reland "[RPC] Use qualified name str directly in RPC torch script code path" ([#None](https://github.com/pytorch/pytorch/pull/None)).
* Reland "[RPC] Use qualified name str directly in RPC torch script code path" (#34962) ([#34962](https://github.com/pytorch/pytorch/pull/34962)).
* [1.5 Release][Dist Autograd][Better Engineering] Notify Workers on Failure during Distributed Autograd (#34638) ([#34638](https://github.com/pytorch/pytorch/pull/34638)).
* [RPC] Add to confirmed users immediately if the fork is shared from owner, instead of adding nothing to pending users (#34988) ([#34988](https://github.com/pytorch/pytorch/pull/34988)).
* Minor fixes for RPC API doc (#34955) ([#34955](https://github.com/pytorch/pytorch/pull/34955)).
* Revert D20442573: [RPC] Use qualified name str directly in RPC torch script code path ([#None](https://github.com/pytorch/pytorch/pull/None)).
* AccumulateGrad: ensure sparse tensor indices and values refcount is always 1 (#34559) ([#34559](https://github.com/pytorch/pytorch/pull/34559)).
