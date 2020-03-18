* Convert KernelTable to a flat-indexed array rather than a hashtable. (#30332) ([#30332](https://github.com/pytorch/pytorch/pull/30332)).
* Remove LeftRight from OperatorEntry and DispatchTable. (#30333) ([#30333](https://github.com/pytorch/pytorch/pull/30333)).
* Make Dispatcher::backendFallbackKernels_ an array (#30340) ([#30340](https://github.com/pytorch/pytorch/pull/30340)).
* Test cases for backend fallback kernels (#29214) ([#29214](https://github.com/pytorch/pytorch/pull/29214)).
* OperatorHandle::callBoxed/callUnboxed (#29330) ([#29330](https://github.com/pytorch/pytorch/pull/29330)).
* better boxing (#29681) ([#29681](https://github.com/pytorch/pytorch/pull/29681)).
* backend fallback test (#29682) ([#29682](https://github.com/pytorch/pytorch/pull/29682)).
* Improve performance of LeftRight::read() (#30282) ([#30282](https://github.com/pytorch/pytorch/pull/30282)).
* Add missing std::move (#30411) ([#30411](https://github.com/pytorch/pytorch/pull/30411)).
* Boxed variable dispatch (#29934) ([#29934](https://github.com/pytorch/pytorch/pull/29934)).
* Use normal dispatch to get to CUDA threshold kernels, instead of DispatchStub. (#30307) ([#30307](https://github.com/pytorch/pytorch/pull/30307)).
* Properly include declaration of dispatch in file that registers it. (#30311) ([#30311](https://github.com/pytorch/pytorch/pull/30311)).
* Remove memory ordering from LeftRight (#31026) ([#31026](https://github.com/pytorch/pytorch/pull/31026)).
* Add private user tensor type IDs for experimentation. (#31830) ([#31830](https://github.com/pytorch/pytorch/pull/31830)).
* Fix the passing-by-ref constructor of OperatorName. (#32170) ([#32170](https://github.com/pytorch/pytorch/pull/32170)).
* Creating callUnboxedWithDispatchKey method (#32198) ([#32198](https://github.com/pytorch/pytorch/pull/32198)).
* Move error reporting code out-of-line from header. (#32118) ([#32118](https://github.com/pytorch/pytorch/pull/32118)).
* Out-of-line construction of OperatorName. (#32121) ([#32121](https://github.com/pytorch/pytorch/pull/32121)).
* Eliminate exception throwing code from dispatch call sites (#32168) ([#32168](https://github.com/pytorch/pytorch/pull/32168)).
* Enhancing the test (#32321) ([#32321](https://github.com/pytorch/pytorch/pull/32321)).
* Enhace DispatchStub to be thread safe from a TSAN point of view. (#32148) ([#32148](https://github.com/pytorch/pytorch/pull/32148)).
* Implement backend fallback fallthrough (#32439) ([#32439](https://github.com/pytorch/pytorch/pull/32439)).
* Improve documentation in dispatcher; remove unnecessary optional (#32533) ([#32533](https://github.com/pytorch/pytorch/pull/32533)).
* Add missing C10_API to dispatch key TLS setter/getters (#32557) ([#32557](https://github.com/pytorch/pytorch/pull/32557)).
* Delete copy/move constructors on these RAII guards. (#32727) ([#32727](https://github.com/pytorch/pytorch/pull/32727)).
* Rename DispatchKey::UndefinedTensorId to Undefined (#32728) ([#32728](https://github.com/pytorch/pytorch/pull/32728)).
* Make DispatchKeyGuards accept DispatchKey::Undefined (#32729) ([#32729](https://github.com/pytorch/pytorch/pull/32729)).
* Centralize addition of "always on" dispatch keys. (#32734) ([#32734](https://github.com/pytorch/pytorch/pull/32734)).
* Stop using dispatchTypeId to do checks for tensor list unwrap. (#32787) ([#32787](https://github.com/pytorch/pytorch/pull/32787)).
* Add XLAPreAutograd key for XLA use cases that need custom autograd. (#32788) ([#32788](https://github.com/pytorch/pytorch/pull/32788)).
* [c10] Allow taking a std::tuple as arg (#32948) ([#32948](https://github.com/pytorch/pytorch/pull/32948)).
* Updated DispatchKeyExtractor to expect TensorOptions (#30981) ([#30981](https://github.com/pytorch/pytorch/pull/30981)).
* Beef up documentation on DispatchKey.h (#33011) ([#33011](https://github.com/pytorch/pytorch/pull/33011)).
* remove dispatch key (#33266) ([#33266](https://github.com/pytorch/pytorch/pull/33266)).
* remove Complex CPU/CUDA backend enum keys (#33267) ([#33267](https://github.com/pytorch/pytorch/pull/33267)).
* Bring up new-style registration API as wrapper around old-style (#33205) ([#33205](https://github.com/pytorch/pytorch/pull/33205)).
* Reduce code duplication in OperatorEntry by keying hash map on optional<DispatchKey> (#33817) ([#33817](https://github.com/pytorch/pytorch/pull/33817)).
* Beef up documentation on Dispatcher.h, reorder methods for clarity. (#33838) ([#33838](https://github.com/pytorch/pytorch/pull/33838)).
* Add a OperatorHandle argument to boxed kernels (#29201) ([#29201](https://github.com/pytorch/pytorch/pull/29201)).
* Hide the OperatorKernel* argument from the stack based kernel API (#29337) ([#29337](https://github.com/pytorch/pytorch/pull/29337)).
* Back out "Back out "Back out "Revert D18542342: Boxed variable dispatch""" (#30650) ([#30650](https://github.com/pytorch/pytorch/pull/30650)).
* Rename TensorTypeId to DispatchKey (#32154) ([#32154](https://github.com/pytorch/pytorch/pull/32154)).
* Use codegen'ed unboxing wrappers (#32521) ([#32521](https://github.com/pytorch/pytorch/pull/32521)).
