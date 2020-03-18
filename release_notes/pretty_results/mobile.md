* Fix exception message in Java Tensor ([#30205](https://github.com/pytorch/pytorch/pull/30205)).
* Fix the crashes for c++ not able to find java class through Jni (#30390) ([#30390](https://github.com/pytorch/pytorch/pull/30390)).
* Add info about transitive dependencies in case of using local aars (#30128) ([#30128](https://github.com/pytorch/pytorch/pull/30128)).
* Add @DoNotStrip to nativeNewTensor method. (#30472) ([#30472](https://github.com/pytorch/pytorch/pull/30472)).
* GenericDict/List type use unshapedType() (#30428) ([#30428](https://github.com/pytorch/pytorch/pull/30428)).
* Add module level qpl logging. (#30906) ([#30906](https://github.com/pytorch/pytorch/pull/30906)).
* Expose setNumThreads to android api (#31033) ([#31033](https://github.com/pytorch/pytorch/pull/31033)).
* Expose setNumThreads to android api (#31205) ([#31205](https://github.com/pytorch/pytorch/pull/31205)).
* Switch to open sourced fbjni (#30175) ([#30175](https://github.com/pytorch/pytorch/pull/30175)).
* Typo in filename align with classname ([#31235](https://github.com/pytorch/pytorch/pull/31235)).
* Loading module from android asset (#30378) ([#30378](https://github.com/pytorch/pytorch/pull/30378)).
* JIT Type parser for mobile (#30391) ([#30391](https://github.com/pytorch/pytorch/pull/30391)).
* Java Tensor hybrid, owns at::Tensor, no memcopy for java outputs. (#30501) ([#30501](https://github.com/pytorch/pytorch/pull/30501)).
* Exclude lite interpreter Java files from OSS host build ([#31204](https://github.com/pytorch/pytorch/pull/31204)).
* Tensor class created from java does not call native methods ([#31520](https://github.com/pytorch/pytorch/pull/31520)).
* Nightly dimension, input shape in gradle (#30195) ([#30195](https://github.com/pytorch/pytorch/pull/30195)).
* Move prim ops from JIT registration to C10 (#30612) ([#30612](https://github.com/pytorch/pytorch/pull/30612)).
* Fix androidTest - exclude host tests from it ([#31522](https://github.com/pytorch/pytorch/pull/31522)).
* Support tensors with a storage offset in Java (#31584) ([#31584](https://github.com/pytorch/pytorch/pull/31584)).
* Update Gemfile (#32147) ([#32147](https://github.com/pytorch/pytorch/pull/32147)).
* Torchscript print to logcat (#31456) ([#31456](https://github.com/pytorch/pytorch/pull/31456)).
* Tensor prep from image in native (#31426) ([#31426](https://github.com/pytorch/pytorch/pull/31426)).
* Add CI scripts for Custom Build (#32316) ([#32316](https://github.com/pytorch/pytorch/pull/32316)).
* Add a new job to support custom build (#32323) ([#32323](https://github.com/pytorch/pytorch/pull/32323)).
* run code analysis against mobile interpreter (#32276) ([#32276](https://github.com/pytorch/pytorch/pull/32276)).
* Set rpath for JNI library on Mac (#32247) ([#32247](https://github.com/pytorch/pytorch/pull/32247)).
* [android] fbjni DoNotStrip annotation for oss native methods (#32567) ([#32567](https://github.com/pytorch/pytorch/pull/32567)).
* Update Docs for building PyTorch for Android. ([#32578](https://github.com/pytorch/pytorch/pull/32578)).
* [pytorch] avoid `thread_local std::vector<Call>` for mobile build (#32849) ([#32849](https://github.com/pytorch/pytorch/pull/32849)).
* Added upsample_neartest2d op for lite interpreter. (#32913) ([#32913](https://github.com/pytorch/pytorch/pull/32913)).
* Add instructions and operators for new bytecode format of PyText model (#33555) ([#33555](https://github.com/pytorch/pytorch/pull/33555)).
* [pytorch][size] remove unused SparseCPUType from mobile build (#33517) ([#33517](https://github.com/pytorch/pytorch/pull/33517)).
* [Lite interpreter] Pass shared_ptr properly (#33667) ([#33667](https://github.com/pytorch/pytorch/pull/33667)).
* Throw an exception when method cannot be found from mobile module. (#33972) ([#33972](https://github.com/pytorch/pytorch/pull/33972)).
* Fix mobile build (#33985) ([#33985](https://github.com/pytorch/pytorch/pull/33985)).
* Fix mobile build (#34000) ([#34000](https://github.com/pytorch/pytorch/pull/34000)).
* [pytorch][mobile] make sure mobile build work with dynamic dispatch (#34038) ([#34038](https://github.com/pytorch/pytorch/pull/34038)).
* [pytorch][mobile] support for custom mobile build with dynamic dispatch (#34055) ([#34055](https://github.com/pytorch/pytorch/pull/34055)).
* Add and test training in lite interpreter. (#32359) ([#32359](https://github.com/pytorch/pytorch/pull/32359)).
* [pytorch][cmake] improve build mobile with host toolchain (#34187) ([#34187](https://github.com/pytorch/pytorch/pull/34187)).
* Add support to dump unsupported ops. Add lite_interpter_load test. (#34072) ([#34072](https://github.com/pytorch/pytorch/pull/34072)).
* [pytorch][CI] add e2e mobile custom build jobs to CI (#34184) ([#34184](https://github.com/pytorch/pytorch/pull/34184)).
* [pytorch] update mobile docker image version (#34337) ([#34337](https://github.com/pytorch/pytorch/pull/34337)).
* Add support to dump unsupported ops. Add lite_interpter_load test. (#34278) ([#34278](https://github.com/pytorch/pytorch/pull/34278)).
* [pytorch][mobile] change mobile build scripts to build PyTorch by default (#34203) ([#34203](https://github.com/pytorch/pytorch/pull/34203)).
* [Lite Trainer] Add necessary registrations for MNIST model (#33717) ([#33717](https://github.com/pytorch/pytorch/pull/33717)).
* [pytorch] fix BUILD_CAFFE2_MOBILE gating around caffe2/operators/experimental/c10/cpu (#34354) ([#34354](https://github.com/pytorch/pytorch/pull/34354)).
* Enable RTTI for mobile builds, to enable custom class via torchbind in mobile (#34368) ([#34368](https://github.com/pytorch/pytorch/pull/34368)).
* [pytorch][ci] add build_only flag to mobile CI jobs (#34560) ([#34560](https://github.com/pytorch/pytorch/pull/34560)).
* [pytorch] remove boilerplate setQEngine() from PyTorch mobile predictors (#34556) ([#34556](https://github.com/pytorch/pytorch/pull/34556)).
* Disable ROCM when building mobile libtorch. (#34478) ([#34478](https://github.com/pytorch/pytorch/pull/34478)).
* [PT] add macro to expose caffe2 ops to PyTorch mobile (#34578) ([#34578](https://github.com/pytorch/pytorch/pull/34578)).
* [JIT][mobile] Support built-in Function call in lite interpreter (#34676) ([#34676](https://github.com/pytorch/pytorch/pull/34676)).
* get rid of choco install (#30897) ([#30897](https://github.com/pytorch/pytorch/pull/30897)).
* Fix SIGABORT caused by double exception in PyTorchStreamReader when file not found. (#33243) ([#33243](https://github.com/pytorch/pytorch/pull/33243)).
* Pass to remove prepacking ops. (#34319) ([#34319](https://github.com/pytorch/pytorch/pull/34319)).
* add irregular c10 op registration/invocation cases to test project (#30558) ([#30558](https://github.com/pytorch/pytorch/pull/30558)).
* add torch_cpu to the static library list in TorchConfig.cmake.in (#30769) ([#30769](https://github.com/pytorch/pytorch/pull/30769)).
* Mobile Backend: NHWC memory layout + XNNPACK integration. (#32509) ([#32509](https://github.com/pytorch/pytorch/pull/32509)).
* Mobile Backend: NHWC memory layout + XNNPACK integration. (#33722) ([#33722](https://github.com/pytorch/pytorch/pull/33722)).
* [Lite Interpreter] Enable __setstate__ (#33294) ([#33294](https://github.com/pytorch/pytorch/pull/33294)).
* Make save_for_lite_interpreter private (#32771) ([#32771](https://github.com/pytorch/pytorch/pull/32771)).
* Use `gettimeofday` on iOS (#30361) ([#30361](https://github.com/pytorch/pytorch/pull/30361)).
* add LLVM-dev package to android docker image (#31215) ([#31215](https://github.com/pytorch/pytorch/pull/31215)).
* Javadoc changes (#31956) ([#31956](https://github.com/pytorch/pytorch/pull/31956)).
* Add File existence checking (#32208) ([#32208](https://github.com/pytorch/pytorch/pull/32208)).
* [pytorch] update code analyzer build.sh to handle srcs with same name (#32525) ([#32525](https://github.com/pytorch/pytorch/pull/32525)).
* Temporarily disable failing iOS builds ([#33154](https://github.com/pytorch/pytorch/pull/33154)).
* [pytorch] convert code analyzer to a binary (#33102) ([#33102](https://github.com/pytorch/pytorch/pull/33102)).
* Fix iOS x86_64 CI failure (#33194) ([#33194](https://github.com/pytorch/pytorch/pull/33194)).
* [iOS] Add watchOS support (#33318) ([#33318](https://github.com/pytorch/pytorch/pull/33318)).
* [iOS][CI] Remove org-member from iOS Simulator Builds (#34410) ([#34410](https://github.com/pytorch/pytorch/pull/34410)).
* Fix SELECTED_OP_LIST file path issue (#33942) ([#33942](https://github.com/pytorch/pytorch/pull/33942)).
* [On-device Benchmark] speed_benchmark_torch switch to log latency from dataset level to row level (#34598) ([#34598](https://github.com/pytorch/pytorch/pull/34598)).
* Fix for handling batch size 0. (#34599) ([#34599](https://github.com/pytorch/pytorch/pull/34599)).
* Integrate XNNPACK with custom class for packing weights. (#34047) ([#34047](https://github.com/pytorch/pytorch/pull/34047)).
* Enable threading for XNNPACK ops. (#34547) ([#34547](https://github.com/pytorch/pytorch/pull/34547)).
* Fix backward compatibility check test for schemas containing (#34782) ([#34782](https://github.com/pytorch/pytorch/pull/34782)).
* [pytorch][mobile] fixed AutoGradMode/AutoNonVariableTypeMode uses for mobile callsites ([#None](https://github.com/pytorch/pytorch/pull/None)).
