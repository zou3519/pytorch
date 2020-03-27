* Turn off scalar_check for masked_select. (#29923) ([#29923](https://github.com/pytorch/pytorch/pull/29923)).
* Make all optimizers consistent so that they don't change gradients inplace ([#30257](https://github.com/pytorch/pytorch/pull/30257)).
* Fix scalar check of MultiLabelMarginLoss. (#30768) ([#30768](https://github.com/pytorch/pytorch/pull/30768)).
* change index_select scalar_check to retain dimensionality of input. (#30790) ([#30790](https://github.com/pytorch/pytorch/pull/30790)).
* MultiMarginCriterion: fix scalar_check in the case where reduction == None. (#30826) ([#30826](https://github.com/pytorch/pytorch/pull/30826)).
* Remove some Half support in some binary CPU kernels (#33021) ([#33021](https://github.com/pytorch/pytorch/pull/33021)).
* Add missing error messages for container modules (#29991) ([#29991](https://github.com/pytorch/pytorch/pull/29991)).
* Remove deprecated codepath for old-style autograd.Function (#30696) (#33956) ([#33956](https://github.com/pytorch/pytorch/pull/33956)).
* End of the .data removal in torch/optim (#34211) ([#34211](https://github.com/pytorch/pytorch/pull/34211)).
* Remove use of `.data` from optimizers (#33640) ([#33640](https://github.com/pytorch/pytorch/pull/33640)).
