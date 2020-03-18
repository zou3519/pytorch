* Add pybind11 trampoline class for c10d.Store (#30415) ([#30415](https://github.com/pytorch/pytorch/pull/30415)).
* Add helper to aggregate multiple process groups (#25768) ([#25768](https://github.com/pytorch/pytorch/pull/25768)).
* Replace deprecated AT_* with TORCH_* to reduce warnings in c10d ([#30795](https://github.com/pytorch/pytorch/pull/30795)).
* DDP should not set grad for globally unused params (#28883) ([#28883](https://github.com/pytorch/pytorch/pull/28883)).
* accept url query when rank or wolrd_size is specified (#32016) ([#32016](https://github.com/pytorch/pytorch/pull/32016)).
* Remove mis-exposed abort API on ProcessGroup ([#32292](https://github.com/pytorch/pytorch/pull/32292)).
* Add ability to abort NCCL communicators from the store. (#32895) ([#32895](https://github.com/pytorch/pytorch/pull/32895)).
* Remove warning about building from source to use the NCCL backend (#34051) ([#34051](https://github.com/pytorch/pytorch/pull/34051)).
* [distributed] pass in timeout to TCP store when initializing (#33325) ([#33325](https://github.com/pytorch/pytorch/pull/33325)).
* Back out "Revert D19871946: [distributed] pass in timeout to TCP store when initializing" (#33434) ([#33434](https://github.com/pytorch/pytorch/pull/33434)).
* transport open registration (#30167) ([#30167](https://github.com/pytorch/pytorch/pull/30167)).
* Fix TCPStoreTest and improve tcputils::connect() (#30354) ([#30354](https://github.com/pytorch/pytorch/pull/30354)).
* Skip ProcessGroupGlooAyncTest if there is no CUDA available (#30345) ([#30345](https://github.com/pytorch/pytorch/pull/30345)).
* abort nccl communicators before throwing operation timed out (#31128) ([#31128](https://github.com/pytorch/pytorch/pull/31128)).
* fix ProcessGroupGlooTest (#31255) ([#31255](https://github.com/pytorch/pytorch/pull/31255)).
* Add missing TORCH_CUDA_API annotation to throw_nccl_error (#31157) ([#31157](https://github.com/pytorch/pytorch/pull/31157)).
* Allow TCPStore to pick a port to bind to. (#31674) ([#31674](https://github.com/pytorch/pytorch/pull/31674)).
* move ProcessGroupGlooTest to gtest (#32133) ([#32133](https://github.com/pytorch/pytorch/pull/32133)).
* Adding DDP Design Note ([#32158](https://github.com/pytorch/pytorch/pull/32158)).
* Add allgather_base as per our discussion re: ProcessGroup interface. (#31892) ([#31892](https://github.com/pytorch/pytorch/pull/31892)).
* add lock for ncclCommAbort (#31901) ([#31901](https://github.com/pytorch/pytorch/pull/31901)).
* fix testSend and testRecv in ProcessGroupGlooTest (#32134) ([#32134](https://github.com/pytorch/pytorch/pull/32134)).
* use gtest asserts in ProcessGroupGlooTest instead of other checks (#32138) ([#32138](https://github.com/pytorch/pytorch/pull/32138)).
* skip testExceptions in ProcessGroupGloo if built with TSAN (#32242) ([#32242](https://github.com/pytorch/pytorch/pull/32242)).
* Enhance NCCL watchdog to acitvely abort communicators for timed out ops. (#32338) ([#32338](https://github.com/pytorch/pytorch/pull/32338)).
* Fix test_data_parallel name errors and add to run_test.py (#32428) ([#32428](https://github.com/pytorch/pytorch/pull/32428)).
* [gloo] Skip registry warning (#31126) ([#31126](https://github.com/pytorch/pytorch/pull/31126)).
* Fix iterator for ncclCommWatchdog. (#32571) ([#32571](https://github.com/pytorch/pytorch/pull/32571)).
* Revert "Fix iterator for ncclCommWatchdog. (#32571)" (#32649) ([#32649](https://github.com/pytorch/pytorch/pull/32649)).
* Fix flaky test_nccl_timeout. (#32653) ([#32653](https://github.com/pytorch/pytorch/pull/32653)).
* [torch] fd error check ([#32797](https://github.com/pytorch/pytorch/pull/32797)).
* Use torch.set_default_dtype in test_data_parallel and rename dtype2prec (#32962) ([#32962](https://github.com/pytorch/pytorch/pull/32962)).
* Fix logging for aborted communicators in ProcessGroupNCCL. (#33147) ([#33147](https://github.com/pytorch/pytorch/pull/33147)).
* [distributed] skip use_ignore_output tests in c10d if not built with gloo (#33513) ([#33513](https://github.com/pytorch/pytorch/pull/33513)).
* [gloo] dont hold locks in calls to buffer in ProcessGroupGloo:RecvWork::wait() and (#33926) ([#33926](https://github.com/pytorch/pytorch/pull/33926)).
* remove duplicated process group gloo timeout (#31342) ([#31342](https://github.com/pytorch/pytorch/pull/31342)).
* fix handling of replica parameters in DataParallel (#33907) ([#33907](https://github.com/pytorch/pytorch/pull/33907)).
* [distributed] quicker exit in the case of failed tests in distributed (#34150) ([#34150](https://github.com/pytorch/pytorch/pull/34150)).
* Add entry for distributed tests to CODEOWNERS. (#34637) ([#34637](https://github.com/pytorch/pytorch/pull/34637)).
* in test_data_parallel.py, remove skipIfRocm from tests that pass (#34978) ([#34978](https://github.com/pytorch/pytorch/pull/34978)).
* [build] Update gloo submodule (#34969) ([#34969](https://github.com/pytorch/pytorch/pull/34969)).
