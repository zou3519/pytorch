* Fix CPU_INTEL flag error on windows (#30564) ([#30564](https://github.com/pytorch/pytorch/pull/30564)).
* Update magma to 2.5.1 for Windows and switch CUDA in CI to 9.2 ([#30513](https://github.com/pytorch/pytorch/pull/30513)).
* Disable implicit conversion warning (#30529) ([#30529](https://github.com/pytorch/pytorch/pull/30529)).
* Don't install pybind11 header directory for system pybind11 installs (#30758) ([#30758](https://github.com/pytorch/pytorch/pull/30758)).
* Upgrade bazel to 1.2.0. (#30885) ([#30885](https://github.com/pytorch/pytorch/pull/30885)).
* Fix conflicts in CMAKE_GENERATOR and generator (#30971) ([#30971](https://github.com/pytorch/pytorch/pull/30971)).
* Fix static cuda builds on older cmake versions (#30935) ([#30935](https://github.com/pytorch/pytorch/pull/30935)).
* For ppc64le, stop presenting the python 2.7 builds (we will no longer… (#32315) ([#32315](https://github.com/pytorch/pytorch/pull/32315)).
* Remove the support of build options like NO_*, WITH_* (#32447) ([#32447](https://github.com/pytorch/pytorch/pull/32447)).
* Make VC++ version a parametrizable option for Windows CI. (#32043) ([#32043](https://github.com/pytorch/pytorch/pull/32043)).
* try to find cudnn header in /usr/include/cuda (#31755) ([#31755](https://github.com/pytorch/pytorch/pull/31755)).
* Patch `Half.h` for compiling CUDA with clang (#29027) ([#29027](https://github.com/pytorch/pytorch/pull/29027)).
* windows template specialization bug (#33076) ([#33076](https://github.com/pytorch/pytorch/pull/33076)).
* Use bazelisk instead of specifying bazel version manually. (#33036) ([#33036](https://github.com/pytorch/pytorch/pull/33036)).
* Drop support of the build option USE_GLOO_IBVERBS (#33163) ([#33163](https://github.com/pytorch/pytorch/pull/33163)).
* Kill old cuda support (#33302) ([#33302](https://github.com/pytorch/pytorch/pull/33302)).
* Update MAGMA to 2.5.2 for Windows (#34205) ([#34205](https://github.com/pytorch/pytorch/pull/34205)).
* Stop using ctypes to interface with CUDA libraries. (#33678) ([#33678](https://github.com/pytorch/pytorch/pull/33678)).
* Mark PyTorch incompatible with python-3.6.0 (#34724) ([#34724](https://github.com/pytorch/pytorch/pull/34724)).
* Don't use RTLD_GLOBAL to load _C. (#31162) ([#31162](https://github.com/pytorch/pytorch/pull/31162)).
* Fix dll load logic for Python 3.8 on Windows (#32215) ([#32215](https://github.com/pytorch/pytorch/pull/32215)).
