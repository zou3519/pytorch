* Remove variable wrapping from register_c10_ops (#29207) ([#29207](https://github.com/pytorch/pytorch/pull/29207)).
* Make the warning of using SparseTensor in JIT less noisy ([#30499](https://github.com/pytorch/pytorch/pull/30499)).
* Add pickler support for Device (#30131) ([#30131](https://github.com/pytorch/pytorch/pull/30131)).
* make sure the counter stays correct in between bailout transitions (#30186) ([#30186](https://github.com/pytorch/pytorch/pull/30186)).
* inline to prevent duplicate obj when linking (#30363) ([#30363](https://github.com/pytorch/pytorch/pull/30363)).
* Dump operator names of a script module (#30467) ([#30467](https://github.com/pytorch/pytorch/pull/30467)).
* Cache compilation of free functions (#30503) ([#30503](https://github.com/pytorch/pytorch/pull/30503)).
* Improve documentation around builtin functions (#30347) ([#30347](https://github.com/pytorch/pytorch/pull/30347)).
* AddConstant and findConstant for ClassType (#29217) ([#29217](https://github.com/pytorch/pytorch/pull/29217)).
* fix anynonzero op ([#29423](https://github.com/pytorch/pytorch/pull/29423)).
* Moving checks related to options.aliasAnalysis and schema.hasAliasInfo to read callsite (#30671) ([#30671](https://github.com/pytorch/pytorch/pull/30671)).
* refactor the way we are handling bailout counts ([#30410](https://github.com/pytorch/pytorch/pull/30410)).
* rename shouldAnnotate api (#30543) ([#30543](https://github.com/pytorch/pytorch/pull/30543)).
* add constant prop for immutable types (#30544) ([#30544](https://github.com/pytorch/pytorch/pull/30544)).
* handle reassignment to inf and nan (#30877) ([#30877](https://github.com/pytorch/pytorch/pull/30877)).
* add tests that schemas are valid (#30749) ([#30749](https://github.com/pytorch/pytorch/pull/30749)).
* polish up overloads on free functions (#30356) ([#30356](https://github.com/pytorch/pytorch/pull/30356)).
* resubmit polish up overloads on free functions (#31014) ([#31014](https://github.com/pytorch/pytorch/pull/31014)).
* Correct definition of nodes that work with Autograd (#30683) ([#30683](https://github.com/pytorch/pytorch/pull/30683)).
* peephole optimize type refinements (#31024) ([#31024](https://github.com/pytorch/pytorch/pull/31024)).
* Make `nn.Transformer` TorchScript compatible (#28561) ([#28561](https://github.com/pytorch/pytorch/pull/28561)).
* Add tracing support for optional Device and Layout (#30979) ([#30979](https://github.com/pytorch/pytorch/pull/30979)).
* Unify list elements for all list types (#30777) ([#30777](https://github.com/pytorch/pytorch/pull/30777)).
* Fix type unification With Specialized Tensor Shapes (#31076) ([#31076](https://github.com/pytorch/pytorch/pull/31076)).
* Resubmit overload v2 (#31123) ([#31123](https://github.com/pytorch/pytorch/pull/31123)).
* fix torch_train build (#30497) ([#30497](https://github.com/pytorch/pytorch/pull/30497)).
* Remove subgraphNode kind assert in unmergeSubgraph (#31212) ([#31212](https://github.com/pytorch/pytorch/pull/31212)).
* Fix handling of type comments in body (#30590) ([#30590](https://github.com/pytorch/pytorch/pull/30590)).
* Introducing ScalarTypeType and LayoutType (#31074) ([#31074](https://github.com/pytorch/pytorch/pull/31074)).
* make profiling take no_grad flags into account (#31071) ([#31071](https://github.com/pytorch/pytorch/pull/31071)).
* Updates to serialization.md (#31372) ([#31372](https://github.com/pytorch/pytorch/pull/31372)).
* Update OVERVIEW.md (#31373) ([#31373](https://github.com/pytorch/pytorch/pull/31373)).
* use expect instead of casting in register_c10_ops (#31401) ([#31401](https://github.com/pytorch/pytorch/pull/31401)).
* remove unnecessary arg from create_script_module (#31017) ([#31017](https://github.com/pytorch/pytorch/pull/31017)).
* remove remnants of properties hack (#31018) ([#31018](https://github.com/pytorch/pytorch/pull/31018)).
* Simplify recursive script compilation flow. (#31019) ([#31019](https://github.com/pytorch/pytorch/pull/31019)).
* avoid doing quadratic work in concrete type inference (#31020) ([#31020](https://github.com/pytorch/pytorch/pull/31020)).
* Move TorchScript language reference to its own page (#31138) ([#31138](https://github.com/pytorch/pytorch/pull/31138)).
* add unsupported section (#31329) ([#31329](https://github.com/pytorch/pytorch/pull/31329)).
* Throw a better error for int too big for int64_t ([#29931](https://github.com/pytorch/pytorch/pull/29931)).
* Fix hex literal parsing (#29935) ([#29935](https://github.com/pytorch/pytorch/pull/29935)).
* Move TorchScript language reference to its own page (#31138) ([#31138](https://github.com/pytorch/pytorch/pull/31138)).
* Cleanup after moving language reference (#31146) ([#31146](https://github.com/pytorch/pytorch/pull/31146)).
* Add support for builtins as attributes (#31269) ([#31269](https://github.com/pytorch/pytorch/pull/31269)).
* add a suggested alternative to _get_trace_graph ([#31441](https://github.com/pytorch/pytorch/pull/31441)).
* Add support for `del` (#31273) ([#31273](https://github.com/pytorch/pytorch/pull/31273)).
* catch all exceptions in converting default values to ivalues (#31398) ([#31398](https://github.com/pytorch/pytorch/pull/31398)).
* fixing a naming issue in creating a residual loop node in a bailout graph (#31400) ([#31400](https://github.com/pytorch/pytorch/pull/31400)).
* Fix lint ([#31463](https://github.com/pytorch/pytorch/pull/31463)).
* Fix builtins table (#31492) ([#31492](https://github.com/pytorch/pytorch/pull/31492)).
* fix missing type check in dictionary literal ([#31375](https://github.com/pytorch/pytorch/pull/31375)).
* Add support for `del` (#31273) ([#31273](https://github.com/pytorch/pytorch/pull/31273)).
* Add fake parsing for torchbind classes in schema type parser ([#31506](https://github.com/pytorch/pytorch/pull/31506)).
* run optimizations on pre-profiled graph (#31392) ([#31392](https://github.com/pytorch/pytorch/pull/31392)).
* Add test to torch.jit.export_opnames, make the _C function private ([#31446](https://github.com/pytorch/pytorch/pull/31446)).
* Support optional float parameters (float?, optional<double>). (#31517) ([#31517](https://github.com/pytorch/pytorch/pull/31517)).
* Fix parsing of big float literals (#29940) ([#29940](https://github.com/pytorch/pytorch/pull/29940)).
* Fix null pointer dereference on Android for strtod_c (#31582) ([#31582](https://github.com/pytorch/pytorch/pull/31582)).
* Bypass _TorchScriptTesting_StackString::pop in BC check now (#31586) ([#31586](https://github.com/pytorch/pytorch/pull/31586)).
* add float[] str[] constants (#31503) ([#31503](https://github.com/pytorch/pytorch/pull/31503)).
* Preserve constant from ConcreteModuleType to ClassType (#29218) ([#29218](https://github.com/pytorch/pytorch/pull/29218)).
* Remove refs from ArrayRef arguments (#31845) ([#31845](https://github.com/pytorch/pytorch/pull/31845)).
* Fix getConstant (#31012) ([#31012](https://github.com/pytorch/pytorch/pull/31012)).
* Add unsafeRemoveConstant for ClassType (#30787) ([#30787](https://github.com/pytorch/pytorch/pull/30787)).
* Don't unconditionally compile runJITCPPTests (#31236) ([#31236](https://github.com/pytorch/pytorch/pull/31236)).
* Fix tracing for modules with List[Tensor] as output (#31343) ([#31343](https://github.com/pytorch/pytorch/pull/31343)).
* Better error for `torch::jit::load`ing a eager file (#31709) ([#31709](https://github.com/pytorch/pytorch/pull/31709)).
* Use real argument names for Python functions (#29300) ([#29300](https://github.com/pytorch/pytorch/pull/29300)).
* Add AliasDb API For Changing Aliasing (#31501) ([#31501](https://github.com/pytorch/pytorch/pull/31501)).
* check for object equality in constant pooling (#31800) ([#31800](https://github.com/pytorch/pytorch/pull/31800)).
* More robust mangling (#31978) ([#31978](https://github.com/pytorch/pytorch/pull/31978)).
* remove .data from torch/jit ([#31480](https://github.com/pytorch/pytorch/pull/31480)).
* Skip un-runnable tests (#31965) ([#31965](https://github.com/pytorch/pytorch/pull/31965)).
* Fix frontend kwarg defualts error (#32146) ([#32146](https://github.com/pytorch/pytorch/pull/32146)).
* Fix an invalid peephole transformation if input/output values are written to (#28455) ([#28455](https://github.com/pytorch/pytorch/pull/28455)).
* Fixes to prim ops (#32179) ([#32179](https://github.com/pytorch/pytorch/pull/32179)).
* dict type unification fix (#32185) ([#32185](https://github.com/pytorch/pytorch/pull/32185)).
* Define `repr()` on IValues (#32232) ([#32232](https://github.com/pytorch/pytorch/pull/32232)).
* classic fixed-point liveness ([#31724](https://github.com/pytorch/pytorch/pull/31724)).
* remove unused code after refactoring optimizations into profiling-sensitive and profiling-insensitive (#32106) ([#32106](https://github.com/pytorch/pytorch/pull/32106)).
* cap the maximum depth of bailout chains at 1 (#32073) ([#32073](https://github.com/pytorch/pytorch/pull/32073)).
* fix unchecked cast alias analysis (#32309) ([#32309](https://github.com/pytorch/pytorch/pull/32309)).
* Remove __torch__ from custom class qualname ([#32301](https://github.com/pytorch/pytorch/pull/32301)).
* Fix returning instance of custom class from method ([#32312](https://github.com/pytorch/pytorch/pull/32312)).
* Test passing custom class instance to bound method ([#32320](https://github.com/pytorch/pytorch/pull/32320)).
* Fix comparisions for ConcreteModuleType (#32256) ([#32256](https://github.com/pytorch/pytorch/pull/32256)).
* Add str[] float[] constants resubmit ([#31791](https://github.com/pytorch/pytorch/pull/31791)).
* improve mayContainAlias (#31839) ([#31839](https://github.com/pytorch/pytorch/pull/31839)).
* remove tuple logic in constant propagation (#31840) ([#31840](https://github.com/pytorch/pytorch/pull/31840)).
* implement tuple constants (#31841) ([#31841](https://github.com/pytorch/pytorch/pull/31841)).
* Corrected logical boolean expression (#32249) ([#32249](https://github.com/pytorch/pytorch/pull/32249)).
* [jit] Enable IValue to hold a PyObject (#32491) ([#32491](https://github.com/pytorch/pytorch/pull/32491)).
* [JIT] throw if no self arg on ignored methods (#32503) ([#32503](https://github.com/pytorch/pytorch/pull/32503)).
* [JIT] Passing custom class as arg (#32260) ([#32260](https://github.com/pytorch/pytorch/pull/32260)).
* [JIT] Test __getstate__ and __setstate__ for custom bound C++ classes ([#32470](https://github.com/pytorch/pytorch/pull/32470)).
* [JIT] Fix custom class method binding for const methods ([#32471](https://github.com/pytorch/pytorch/pull/32471)).
* [JIT] Support returning tuple from custom bound C++ method ([#32477](https://github.com/pytorch/pytorch/pull/32477)).
* Revert "Remove __torch__ from custom class qualname" (#32514) ([#32514](https://github.com/pytorch/pytorch/pull/32514)).
* [JIT] Remove capsule type handling of node hashing (#32540) ([#32540](https://github.com/pytorch/pytorch/pull/32540)).
* [jit] Fix dict type serialization (#32569) ([#32569](https://github.com/pytorch/pytorch/pull/32569)).
* API for testing bailouts (#32518) ([#32518](https://github.com/pytorch/pytorch/pull/32518)).
* [jit] allow compilation using optional modules (#32539) ([#32539](https://github.com/pytorch/pytorch/pull/32539)).
* [docs] Change fut.wait() to torch.jit._wait(fut) in jit overview docs (#32336) ([#32336](https://github.com/pytorch/pytorch/pull/32336)).
* [jit] Cloning constants in ClassType (#32371) ([#32371](https://github.com/pytorch/pytorch/pull/32371)).
* faster bailout tests (#32266) ([#32266](https://github.com/pytorch/pytorch/pull/32266)).
* [JIT] Fix classes as attributes in recursive scripting ([#32594](https://github.com/pytorch/pytorch/pull/32594)).
* [jit] fix segfault on missing getstate (#32642) ([#32642](https://github.com/pytorch/pytorch/pull/32642)).
* Python binding to export bytecode format for lite interpreter (#32621) ([#32621](https://github.com/pytorch/pytorch/pull/32621)).
* consider FAIL_GUARD while counting indices for GUARDs (#32672) ([#32672](https://github.com/pytorch/pytorch/pull/32672)).
* [JIT] Support for registering C++ lambdas as methods on custom C++ class ([#32553](https://github.com/pytorch/pytorch/pull/32553)).
* [JIT] Serialize attributes and types in ClassType serialization ([#32555](https://github.com/pytorch/pytorch/pull/32555)).
* [JIT] Improve May Contain Alias Using Contained Elements (#32326) ([#32326](https://github.com/pytorch/pytorch/pull/32326)).
* fix windows build (#32762) ([#32762](https://github.com/pytorch/pytorch/pull/32762)).
* [jit] fix the NoneType param/buffer hack (#32745) ([#32745](https://github.com/pytorch/pytorch/pull/32745)).
* Fix/simplify alias annotation handling in op codegen. (#32574) ([#32574](https://github.com/pytorch/pytorch/pull/32574)).
* Add knobs to set the number of profiling runs and bailout depth (#32735) ([#32735](https://github.com/pytorch/pytorch/pull/32735)).
* Code cleaning: Some iterating variables in builtin_functions.cpp can be const (#32852) ([#32852](https://github.com/pytorch/pytorch/pull/32852)).
* [JIT] namedtuple constants (#32873) ([#32873](https://github.com/pytorch/pytorch/pull/32873)).
* [JIT] fix nested select assign (#32877) ([#32877](https://github.com/pytorch/pytorch/pull/32877)).
* [PyTorch][TorchScript] Add support for join on List of strings in TorchScript (#32847) ([#32847](https://github.com/pytorch/pytorch/pull/32847)).
* [JIT] make is_scripting a condvalue (#32871) ([#32871](https://github.com/pytorch/pytorch/pull/32871)).
* Adding scalar to the c10 registration type check ([#32886](https://github.com/pytorch/pytorch/pull/32886)).
* Clarify the searched string is displayed in the error message ([#32789](https://github.com/pytorch/pytorch/pull/32789)).
* [JIT] Update OVERVIEW.md ([#28870](https://github.com/pytorch/pytorch/pull/28870)).
* raise when jit-load.ing a folder (#27836) ([#27836](https://github.com/pytorch/pytorch/pull/27836)).
* Remove unneded TORCH_API (#32015) ([#32015](https://github.com/pytorch/pytorch/pull/32015)).
* [JIT] Trace uses of torchbind classes as module attributes (#32833) ([#32833](https://github.com/pytorch/pytorch/pull/32833)).
* [JIT] Make IRParser use op schema (#32854) ([#32854](https://github.com/pytorch/pytorch/pull/32854)).
* [JIT] Fix python pickle serialization for torchbind (#32878) ([#32878](https://github.com/pytorch/pytorch/pull/32878)).
* Attempt to workaround MSVC17 static constexpr bug ([#33002](https://github.com/pytorch/pytorch/pull/33002)).
* [jit] fix parser for one-line functions (#32941) ([#32941](https://github.com/pytorch/pytorch/pull/32941)).
* Fix some bugs with zipfile serialization (#32244) ([#32244](https://github.com/pytorch/pytorch/pull/32244)).
* [JIT] Resolve custom classes in source importer (#32977) ([#32977](https://github.com/pytorch/pytorch/pull/32977)).
* [JIT] Add Exit Transform / Convert To SSA to docs ([#24114](https://github.com/pytorch/pytorch/pull/24114)).
* [jit] Minor: avoid recalculating some keys for map accesses in pickler. (#33060) ([#33060](https://github.com/pytorch/pytorch/pull/33060)).
* move Decompose before profiling to prevent clearing shape info ([#33100](https://github.com/pytorch/pytorch/pull/33100)).
* [pytorch] Elide more Thrift Tensor send copies. (#31998) ([#31998](https://github.com/pytorch/pytorch/pull/31998)).
* [jit] fix a typo ([#29107](https://github.com/pytorch/pytorch/pull/29107)).
* Remove Node dependencies from operator.h (#32682) ([#32682](https://github.com/pytorch/pytorch/pull/32682)).
* remove unnecessary Node* ops (#32760) ([#32760](https://github.com/pytorch/pytorch/pull/32760)).
* Remove ImplicitTensorToNum (#32761) ([#32761](https://github.com/pytorch/pytorch/pull/32761)).
* [jit] Support properties on `Device` (#32953) ([#32953](https://github.com/pytorch/pytorch/pull/32953)).
* [JIT] peephole optimize values with NoneType (#33264) ([#33264](https://github.com/pytorch/pytorch/pull/33264)).
* [jit] Add RRef to IValue and JIT type system (#32992) ([#32992](https://github.com/pytorch/pytorch/pull/32992)).
* Simplify prim::shape when we have complete tensor types. (#33336) ([#33336](https://github.com/pytorch/pytorch/pull/33336)).
* Allow to register custom passes both before and after fusion. (#33261) ([#33261](https://github.com/pytorch/pytorch/pull/33261)).
* Add support for aten::slice to guard elimination. (#33311) ([#33311](https://github.com/pytorch/pytorch/pull/33311)).
* [Fuser] Add a knob for disabling/enabling CUDA fuser. (#33395) ([#33395](https://github.com/pytorch/pytorch/pull/33395)).
* Fix avx-512 detection logic for jit fuser with MSVC 2019 (#33403) ([#33403](https://github.com/pytorch/pytorch/pull/33403)).
* Add guard elimination support for aten::unsqueeze. (#33371) ([#33371](https://github.com/pytorch/pytorch/pull/33371)).
* interpreter handling for varargs to remove need for looking at Node (#32791) ([#32791](https://github.com/pytorch/pytorch/pull/32791)).
* Remove prim::Constant op (#32804) ([#32804](https://github.com/pytorch/pytorch/pull/32804)).
* [jit] Delete the ErrorReport default constructor (#32879) ([#32879](https://github.com/pytorch/pytorch/pull/32879)).
* [jit] de-optionalize SourceRange context (#32880) ([#32880](https://github.com/pytorch/pytorch/pull/32880)).
* [jit] Remove `torch.jit._dump_trace (#33453) ([#33453](https://github.com/pytorch/pytorch/pull/33453)).
* [jit][fix] Remove slot in parameter slot (#32846) ([#32846](https://github.com/pytorch/pytorch/pull/32846)).
* [JIT] Add more ops to 'removableGuard' in guard elimination pass. (#33465) ([#33465](https://github.com/pytorch/pytorch/pull/33465)).
* [jit] Fix ModuleDict type sharing (#33515) ([#33515](https://github.com/pytorch/pytorch/pull/33515)).
* [jit] add `inlined_graph` method to ScriptFunctions (#33508) ([#33508](https://github.com/pytorch/pytorch/pull/33508)).
* [jit] Add None parameter as parameter instead of attributes (#32964) ([#32964](https://github.com/pytorch/pytorch/pull/32964)).
* run peephole to do profile-based optimizations (#33337) ([#33337](https://github.com/pytorch/pytorch/pull/33337)).
* strict check for a device type in Fuser (#33025) ([#33025](https://github.com/pytorch/pytorch/pull/33025)).
* Replace AT_CHECK with TORCH_CHECK in torch/csrc/jit/pybind_utils.h (#33524) ([#33524](https://github.com/pytorch/pytorch/pull/33524)).
* [jit] Fix aug assign for non-tensor attributes (#32993) ([#32993](https://github.com/pytorch/pytorch/pull/32993)).
* refactor strongTypePtr (#33590) ([#33590](https://github.com/pytorch/pytorch/pull/33590)).
* Fix bug where we were trying to get a schema for prim::Constant, which is not registered as an operator. (#33645) ([#33645](https://github.com/pytorch/pytorch/pull/33645)).
* catch and propagate warnings for JIT ScriptMethods (#33010) ([#33010](https://github.com/pytorch/pytorch/pull/33010)).
* [docs] add experimental warning to TorchScript classes in language reference (#33697) ([#33697](https://github.com/pytorch/pytorch/pull/33697)).
* [jit] Unify augmented assign handling (#33578) ([#33578](https://github.com/pytorch/pytorch/pull/33578)).
* add support for _modules, reducing special casing of nn.Sequential (#29495) ([#29495](https://github.com/pytorch/pytorch/pull/29495)).
* [JIT] add support for torch.lu to torchscript (#33724) ([#33724](https://github.com/pytorch/pytorch/pull/33724)).
* [JIT] add support for lu_unpack (#33736) ([#33736](https://github.com/pytorch/pytorch/pull/33736)).
* [JIT] add support for torch.cdist (#33737) ([#33737](https://github.com/pytorch/pytorch/pull/33737)).
* [jit] Unify augmented assign handling (#33578) ([#33578](https://github.com/pytorch/pytorch/pull/33578)).
* [JIT] Support calling Tensor.element_size() in TorchScript (#33808) ([#33808](https://github.com/pytorch/pytorch/pull/33808)).
* fix lint (#33861) ([#33861](https://github.com/pytorch/pytorch/pull/33861)).
* [JIT] Introduce a fake Tensor creation node for IR unit tests (#33595) ([#33595](https://github.com/pytorch/pytorch/pull/33595)).
* [jit] remove some unused/redundant files (#33806) ([#33806](https://github.com/pytorch/pytorch/pull/33806)).
* [jit] add top-level readme to csrc/jit (#33916) ([#33916](https://github.com/pytorch/pytorch/pull/33916)).
* [jit] fix up refs in overview.md (#33919) ([#33919](https://github.com/pytorch/pytorch/pull/33919)).
* [JIT] Implement Tensor.tolist() (#33472) ([#33472](https://github.com/pytorch/pytorch/pull/33472)).
* [jit] Resolve type annotation names to types (#29623) ([#29623](https://github.com/pytorch/pytorch/pull/29623)).
* [jit] Add missing tensor properties (#33906) ([#33906](https://github.com/pytorch/pytorch/pull/33906)).
* [JIT] fix alias assertion (#33952) ([#33952](https://github.com/pytorch/pytorch/pull/33952)).
* Make HashNode visible (#34045) ([#34045](https://github.com/pytorch/pytorch/pull/34045)).
* Fix typo (#33925) ([#33925](https://github.com/pytorch/pytorch/pull/33925)).
* improved TorchScript traceback (#33834) ([#33834](https://github.com/pytorch/pytorch/pull/33834)).
* [JIT] Add modulelist indexing for integer literal (#29236) ([#29236](https://github.com/pytorch/pytorch/pull/29236)).
* [JIT] fix alias assertion (#34268) ([#34268](https://github.com/pytorch/pytorch/pull/34268)).
* Throw a proper error when parsing local variable annotations without assignments (#34133) ([#34133](https://github.com/pytorch/pytorch/pull/34133)).
* [JIT] Introduce a fake Tensor creation node for IR unit tests (#33914) ([#33914](https://github.com/pytorch/pytorch/pull/33914)).
* [JIT] add support for torch.norm (#33783) ([#33783](https://github.com/pytorch/pytorch/pull/33783)).
* [JIT] Add support for list() (#33818) ([#33818](https://github.com/pytorch/pytorch/pull/33818)).
* [JIT] add other module apis (#34106) ([#34106](https://github.com/pytorch/pytorch/pull/34106)).
* [aten] Don't deadlock in IValue::Future impl, tests. (#34099) ([#34099](https://github.com/pytorch/pytorch/pull/34099)).
* [JIT] Move stuff out of class_type.cpp (#33900) ([#33900](https://github.com/pytorch/pytorch/pull/33900)).
* [jit] Make `ModuleList`s a sugared value (#34320) ([#34320](https://github.com/pytorch/pytorch/pull/34320)).
* [JIT] remove list with default builtin (#34171) ([#34171](https://github.com/pytorch/pytorch/pull/34171)).
* Dictionary Constants (#32869) ([#32869](https://github.com/pytorch/pytorch/pull/32869)).
* profile block outputs; helps guard elimination (#33889) ([#33889](https://github.com/pytorch/pytorch/pull/33889)).
* [JIT] Preserve qualified names on traced modules (#34395) ([#34395](https://github.com/pytorch/pytorch/pull/34395)).
* [JIT] Introduce a fake Tensor creation node for IR unit tests (#34334) ([#34334](https://github.com/pytorch/pytorch/pull/34334)).
* Delete OperatorOptions, absorb AliasAnalysisKind into FunctionSchema. (#34160) ([#34160](https://github.com/pytorch/pytorch/pull/34160)).
* [JIT] Torchbind error if python instantiate class that doesnt exist (#34568) ([#34568](https://github.com/pytorch/pytorch/pull/34568)).
* [JIT] Add support for tolist for GPU-resident Tensors (#34554) ([#34554](https://github.com/pytorch/pytorch/pull/34554)).
* Delete OperatorOptions, absorb AliasAnalysisKind into FunctionSchema. (#34588) ([#34588](https://github.com/pytorch/pytorch/pull/34588)).
* [JIT] remove specialized list ops (#34520) ([#34520](https://github.com/pytorch/pytorch/pull/34520)).
* Attempt to resolve inconsistent dll linkage warnings on MSVC (#34639) ([#34639](https://github.com/pytorch/pytorch/pull/34639)).
* [JIT] EliminateDeadCode shouldn't remove custom operator node that has untracked mutation (#34635) ([#34635](https://github.com/pytorch/pytorch/pull/34635)).
* [JIT] remove specialized list ops (#34520) ([#34520](https://github.com/pytorch/pytorch/pull/34520)).
* Support left and right shift operators in JIT (#34563) ([#34563](https://github.com/pytorch/pytorch/pull/34563)).
* invokeOperatorFromPython: support overloaded operator calling (#34671) ([#34671](https://github.com/pytorch/pytorch/pull/34671)).
* Move torchbind out of jit namespace (#34745) ([#34745](https://github.com/pytorch/pytorch/pull/34745)).
* Add overloaded name to prim operators (#34280) ([#34280](https://github.com/pytorch/pytorch/pull/34280)).
* [jit] copy unused/ignored methods to ScriptModule during compilation (#33981) ([#33981](https://github.com/pytorch/pytorch/pull/33981)).
* [torchbind] Test moving custom classes to/from IValue (#34847) ([#34847](https://github.com/pytorch/pytorch/pull/34847)).
* [TensorExpr] Pull changes from bertmaher/pytorch_fusion. (#34842) ([#34842](https://github.com/pytorch/pytorch/pull/34842)).
* Eliminate guards through max_pool ops. (#34512) ([#34512](https://github.com/pytorch/pytorch/pull/34512)).
* Format register_ditributed_ops.cpp (#34922) ([#34922](https://github.com/pytorch/pytorch/pull/34922)).
* Make checkInputs more robust (#34838) ([#34838](https://github.com/pytorch/pytorch/pull/34838)).
* [torchbind] Improve IValue custom class API and remove most Capsule stuff (#34848) ([#34848](https://github.com/pytorch/pytorch/pull/34848)).
* (de)serialization of values between C++ and Python (#30108) ([#30108](https://github.com/pytorch/pytorch/pull/30108)).
* Fix getAttribute (#31011) ([#31011](https://github.com/pytorch/pytorch/pull/31011)).
* Expose class constant through `attr` and `setattr` in object (#29219) ([#29219](https://github.com/pytorch/pytorch/pull/29219)).
* Retain the order of parameters while generating ConcreteModuleTypes (#34131) ([#34131](https://github.com/pytorch/pytorch/pull/34131)).
* remove list specialization from ivalue (#30734) ([#30734](https://github.com/pytorch/pytorch/pull/30734)).
* Renaming IValue List functions (#32093) ([#32093](https://github.com/pytorch/pytorch/pull/32093)).
* Move special ops into interpreter (#32889) ([#32889](https://github.com/pytorch/pytorch/pull/32889)).
* Clean up isinstance flags (#33265) ([#33265](https://github.com/pytorch/pytorch/pull/33265)).
* [jit] Add type tags to lists/dicts in pickle (#33255) ([#33255](https://github.com/pytorch/pytorch/pull/33255)).
* Back out "[jit] Add type tags to lists/dicts in pickle" (#34406) ([#34406](https://github.com/pytorch/pytorch/pull/34406)).
* [jit] Add type tags to lists/dicts in pickle (#33255) ([#33255](https://github.com/pytorch/pytorch/pull/33255)).
* Detect TorchScript archives in torch.load (#29339) ([#29339](https://github.com/pytorch/pytorch/pull/29339)).
* Stop producing op_version_set version numbers. ([#28122](https://github.com/pytorch/pytorch/pull/28122)).
* Separate torchbind from Python (#30242) ([#30242](https://github.com/pytorch/pytorch/pull/30242)).
* [JIT] pickle serialization for custom bound classes ([#32604](https://github.com/pytorch/pytorch/pull/32604)).
* [JIT] Use Type Level Granularity in Alias Analysis Wildcards (#32251) ([#32251](https://github.com/pytorch/pytorch/pull/32251)).
* Rename TorchScript compiler to IR emitter to better reflect its function. (#33127) ([#33127](https://github.com/pytorch/pytorch/pull/33127)).
* [TensorExpr] Add classes for memory management in tensor expressions. (#33216) ([#33216](https://github.com/pytorch/pytorch/pull/33216)).
* [TensorExpr] Add a class for representing data type. (#33217) ([#33217](https://github.com/pytorch/pytorch/pull/33217)).
* [TensorExpr] Add core classes for representing expressions and statements. (#33218) ([#33218](https://github.com/pytorch/pytorch/pull/33218)).
* [TensorExpr] Add IR visitor, IR mutator, and IR evaluator. (#33219) ([#33219](https://github.com/pytorch/pytorch/pull/33219)).
* [TensorExpr] Add IR Printer. (#33220) ([#33220](https://github.com/pytorch/pytorch/pull/33220)).
* [TensorExpr] Add a boilerplate pass for future TensorExpr fusion pass. (#33464) ([#33464](https://github.com/pytorch/pytorch/pull/33464)).
* Freezing Torchscript modules (#32178) ([#32178](https://github.com/pytorch/pytorch/pull/32178)).
* [jit] Fix iOS build (#34180) ([#34180](https://github.com/pytorch/pytorch/pull/34180)).
* [CUDA_FUSER] Fork CUDA fuser (#33527) ([#33527](https://github.com/pytorch/pytorch/pull/33527)).
* [JIT] Introduce BuiltinOpFunction and integrate into torchbind (#34098) ([#34098](https://github.com/pytorch/pytorch/pull/34098)).
* [jit] delete netdef converter (#33807) ([#33807](https://github.com/pytorch/pytorch/pull/33807)).
* Enable JIT tests on Windows (#27029) ([#27029](https://github.com/pytorch/pytorch/pull/27029)).
* [TensorExpr] Pull changes to core classes for representing expressions and statements from the side branch. (#34224) ([#34224](https://github.com/pytorch/pytorch/pull/34224)).
* [TensorExpr] Add a fuser pass based on tensor expressions. (#34226) ([#34226](https://github.com/pytorch/pytorch/pull/34226)).
* [TensorExpr] Add CUDA codegen. (#34227) ([#34227](https://github.com/pytorch/pytorch/pull/34227)).
* [TensorExpr] Add LLVM codegen. (#34228) ([#34228](https://github.com/pytorch/pytorch/pull/34228)).
* [jit] do the code reorg (#33851) ([#33851](https://github.com/pytorch/pytorch/pull/33851)).
* [JIT] Virtualize Function (#33921) ([#33921](https://github.com/pytorch/pytorch/pull/33921)).
* [jit] kill script namespace (#34515) ([#34515](https://github.com/pytorch/pytorch/pull/34515)).
* [jit] Module clone work with shared ClassType (#31970) ([#31970](https://github.com/pytorch/pytorch/pull/31970)).
* Remove stray `@script` (#32235) ([#32235](https://github.com/pytorch/pytorch/pull/32235)).
* [JIT] fix resolving of functions in torch/functional. fix compilation of torch.stft (#33504) ([#33504](https://github.com/pytorch/pytorch/pull/33504)).
* [JIT] remove builtin interpolate functions (#34514) ([#34514](https://github.com/pytorch/pytorch/pull/34514)).
* [jit] small cleanups after script:: removal (#34677) ([#34677](https://github.com/pytorch/pytorch/pull/34677)).
* [jit] stop printing crap in test_jit (#33779) ([#33779](https://github.com/pytorch/pytorch/pull/33779)).
* [jit] stop printing crap in test_jit (#33917) ([#33917](https://github.com/pytorch/pytorch/pull/33917)).
* add a warning for script classes (#31069) ([#31069](https://github.com/pytorch/pytorch/pull/31069)).
* remove / rewrite weak module tests (#31193) ([#31193](https://github.com/pytorch/pytorch/pull/31193)).
* Use `default_observer` and `default_weight_observer` in tests (#31424) ([#31424](https://github.com/pytorch/pytorch/pull/31424)).
* add back in reference to jit_unsupported section (#31486) ([#31486](https://github.com/pytorch/pytorch/pull/31486)).
* Add Python language reference docs (#30686) ([#30686](https://github.com/pytorch/pytorch/pull/30686)).
* Document `IValue` (#31904) ([#31904](https://github.com/pytorch/pytorch/pull/31904)).
* temporary fix for jit test backward compatibility issues ([#31949](https://github.com/pytorch/pytorch/pull/31949)).
* Remove temporary fix for torchbind in BC check (#31982) ([#31982](https://github.com/pytorch/pytorch/pull/31982)).
* Temporary workaround for BC test due to schema parser changes ([#32324](https://github.com/pytorch/pytorch/pull/32324)).
* Fix BC test after TorchBind cahnges (#32429) ([#32429](https://github.com/pytorch/pytorch/pull/32429)).
* Revert "Temporary workaround for BC test due to schema parser changes" (#32441) ([#32441](https://github.com/pytorch/pytorch/pull/32441)).
* Add unit test on export_opnames with interface. (#31531) ([#31531](https://github.com/pytorch/pytorch/pull/31531)).
* [JIT] Add torch.classes.load_library ([#32508](https://github.com/pytorch/pytorch/pull/32508)).
* [JIT] Include custom_class.h in torch/script.h ([#32586](https://github.com/pytorch/pytorch/pull/32586)).
* [JIT] Fix stateful lambda stuff and simplify code in custom C++ binding API ([#32658](https://github.com/pytorch/pytorch/pull/32658)).
* Use direct vector indexing in Object::getSlot() instead of at(). (#31627) ([#31627](https://github.com/pytorch/pytorch/pull/31627)).
* replaces .at with [] in getSlot (#32677) ([#32677](https://github.com/pytorch/pytorch/pull/32677)).
* [jit] remove redundant variables from JIT TestCase ([#29091](https://github.com/pytorch/pytorch/pull/29091)).
* [JIT] remove inline everything jitter skip (#33468) ([#33468](https://github.com/pytorch/pytorch/pull/33468)).
* Use cheaper check in isTensorList (#33528) ([#33528](https://github.com/pytorch/pytorch/pull/33528)).
* add bailout checks to checkScript (#32802) ([#32802](https://github.com/pytorch/pytorch/pull/32802)).
* [JIT] Fix FunctionType::python_str() (#33680) ([#33680](https://github.com/pytorch/pytorch/pull/33680)).
* [jit] Fix flipped PackedSequence outputs in script (#32955) ([#32955](https://github.com/pytorch/pytorch/pull/32955)).
* Check fuser results when profiling (#33944) ([#33944](https://github.com/pytorch/pytorch/pull/33944)).
* [jit] Fix flipped PackedSequence outputs in script (#32955) ([#32955](https://github.com/pytorch/pytorch/pull/32955)).
* Make tracing in code gen optional (#33715) ([#33715](https://github.com/pytorch/pytorch/pull/33715)).
* [JIT] disable test (#34722) ([#34722](https://github.com/pytorch/pytorch/pull/34722)).
* Remove unnecessary import (#34778) ([#34778](https://github.com/pytorch/pytorch/pull/34778)).
* [TensorExpr] Add tensorexpr benchmarks. (#34230) ([#34230](https://github.com/pytorch/pytorch/pull/34230)).
* [testing][do not land] (#34605) ([#34605](https://github.com/pytorch/pytorch/pull/34605)).
* [torchbind] Add more comprehensive docscrings (#34906) ([#34906](https://github.com/pytorch/pytorch/pull/34906)).
* Doxygen for torchbind (#35007) ([#35007](https://github.com/pytorch/pytorch/pull/35007)).
* Fix warnings in test/test_jit_fuser.py (#34980) ([#34980](https://github.com/pytorch/pytorch/pull/34980)).
* port ge changes from bert/pytorch_fusion (#34942) ([#34942](https://github.com/pytorch/pytorch/pull/34942)).
* [jit] Include call stack in OSError message (#34669) ([#34669](https://github.com/pytorch/pytorch/pull/34669)).
* Delete unnecessary aliasAnalysis specification from operator registrations. (#33093) ([#33093](https://github.com/pytorch/pytorch/pull/33093)).
