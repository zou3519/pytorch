* Try exporting ONNX with force_outplace=False (#29466) ([#29466](https://github.com/pytorch/pytorch/pull/29466)).
* Provide names for operator nodes in ONNX exported graph. (#27342) ([#27342](https://github.com/pytorch/pytorch/pull/27342)).
* Export dynamic unbind/split and __getitem__ (#29136) ([#29136](https://github.com/pytorch/pytorch/pull/29136)).
* Enable constant folding (#29834) ([#29834](https://github.com/pytorch/pytorch/pull/29834)).
* Support logsoftmax with dim != -1 (#30433) ([#30433](https://github.com/pytorch/pytorch/pull/30433)).
* Add ONNX Scripting Conv Support (#30618) ([#30618](https://github.com/pytorch/pytorch/pull/30618)).
* Add JIT pass to insert permutes for conv ops (#30679) ([#30679](https://github.com/pytorch/pytorch/pull/30679)).
* Support exporting aten::copy_ and aten::index_put to ONNX opset 11 (#26941) ([#26941](https://github.com/pytorch/pytorch/pull/26941)).
* Export custom ops (#29752) ([#29752](https://github.com/pytorch/pytorch/pull/29752)).
* Add quantized concat conversion (#30887) ([#30887](https://github.com/pytorch/pytorch/pull/30887)).
* Fix weight_norm export for dim=0 (#31015) ([#31015](https://github.com/pytorch/pytorch/pull/31015)).
* Add ONNX Export Support to floor_divide (#31081) ([#31081](https://github.com/pytorch/pytorch/pull/31081)).
* ONNX Interpolate Add Scales Params (#28324) ([#28324](https://github.com/pytorch/pytorch/pull/28324)).
* Partially support tensor lists in loop/concat/stack (#30126) ([#30126](https://github.com/pytorch/pytorch/pull/30126)).
* Update ONNX Flatten to accept negative indices in opset 11 (#30751) ([#30751](https://github.com/pytorch/pytorch/pull/30751)).
* Upgrade exported ONNX IR version to 6 (#31025) ([#31025](https://github.com/pytorch/pytorch/pull/31025)).
* Im2col export (#30972) ([#30972](https://github.com/pytorch/pytorch/pull/30972)).
* Renaming scales parameter for interpolate (#31526) ([#31526](https://github.com/pytorch/pytorch/pull/31526)).
* Remove non-ascii character from torch/onnx/symbolic_opset11.py ([#31814](https://github.com/pytorch/pytorch/pull/31814)).
* Added support for Dim operation in ONNX export (#31928) ([#31928](https://github.com/pytorch/pytorch/pull/31928)).
* example_outputs Doc Edit (#31826) ([#31826](https://github.com/pytorch/pytorch/pull/31826)).
* Support op registration if name starts with underscore (_) (#32017) ([#32017](https://github.com/pytorch/pytorch/pull/32017)).
* Sort export w/ negative axes (#31971) ([#31971](https://github.com/pytorch/pytorch/pull/31971)).
* Added cons folding for ONNX mul, div, sqrt ops (#32077) ([#32077](https://github.com/pytorch/pytorch/pull/32077)).
* Added ONNX model checker to ONNX export (#32298) ([#32298](https://github.com/pytorch/pytorch/pull/32298)).
* Fixed access to element in size tensor for scripting (#32652) ([#32652](https://github.com/pytorch/pytorch/pull/32652)).
* [ONNX] Update ONNX landing page since 1.3 (#32805) ([#32805](https://github.com/pytorch/pytorch/pull/32805)).
* [ONNX] Add einsum export (#32716) ([#32716](https://github.com/pytorch/pytorch/pull/32716)).
* [ONNX] Update support of exporting bool type index mask (#32445) ([#32445](https://github.com/pytorch/pytorch/pull/32445)).
* [ONNX] Fix for constant folding flaky tests (#32546) ([#32546](https://github.com/pytorch/pytorch/pull/32546)).
* [ONNX] Export bitwise_not for bool (logical_not) (#28439) ([#28439](https://github.com/pytorch/pytorch/pull/28439)).
* Interpolate Float [] support in ONNX (#32554) ([#32554](https://github.com/pytorch/pytorch/pull/32554)).
* [ONNX] Add flag to enable script tests (#32654) ([#32654](https://github.com/pytorch/pytorch/pull/32654)).
* [ONNX] Fix exporting copy_ with index as tensor input (#32801) ([#32801](https://github.com/pytorch/pytorch/pull/32801)).
* Modifed randNLike for scripting (#32830) ([#32830](https://github.com/pytorch/pytorch/pull/32830)).
* [ONNX] Extend op registration to next opsets (#32943) ([#32943](https://github.com/pytorch/pytorch/pull/32943)).
* ONNX support for torch.take (#33061) ([#33061](https://github.com/pytorch/pytorch/pull/33061)).
* Support using scalar tensor for split (#32493) ([#32493](https://github.com/pytorch/pytorch/pull/32493)).
* [ONNX] Fix export for avg_pool with default stride (#33017) ([#33017](https://github.com/pytorch/pytorch/pull/33017)).
* [ONNX] Fix ONNX CI (#33200) ([#33200](https://github.com/pytorch/pytorch/pull/33200)).
* Fix for rand_like as well. (#33095) ([#33095](https://github.com/pytorch/pytorch/pull/33095)).
* [ONNX] Skip problematic ONNX test to unblock CI (#33323) ([#33323](https://github.com/pytorch/pytorch/pull/33323)).
* [ONNX] Export split with list of sizes (#33161) ([#33161](https://github.com/pytorch/pytorch/pull/33161)).
* [ONNX] Adding ONNX large model export support in exporter (#33062) ([#33062](https://github.com/pytorch/pytorch/pull/33062)).
* Updating ONNX checker logic. (#33522) ([#33522](https://github.com/pytorch/pytorch/pull/33522)).
* Turn ONNX_ML into a proper build option. (#33424) ([#33424](https://github.com/pytorch/pytorch/pull/33424)).
* ONNX Error Message on Missing Op (#33593) ([#33593](https://github.com/pytorch/pytorch/pull/33593)).
* ONNX Export Support for NLLLoss (#33509) ([#33509](https://github.com/pytorch/pytorch/pull/33509)).
* [ONNX] Reduce ONNX test time on CI (#33242) ([#33242](https://github.com/pytorch/pytorch/pull/33242)).
* [ONNX] Fix for random generators export (#33789) ([#33789](https://github.com/pytorch/pytorch/pull/33789)).
* [ONNX] Export new_zeros (#34077) ([#34077](https://github.com/pytorch/pytorch/pull/34077)).
* [ONNX] Support one_hot (#34454) ([#34454](https://github.com/pytorch/pytorch/pull/34454)).
* small typos (#34589) ([#34589](https://github.com/pytorch/pytorch/pull/34589)).
* [quant][onnx] Add support to convert max_pool2d quantized pytorch op to C2 (#33945) ([#33945](https://github.com/pytorch/pytorch/pull/33945)).
* [quant][onnx] Support conversion of quantized sigmoid operator from pytorch to caffe2 (#34629) ([#34629](https://github.com/pytorch/pytorch/pull/34629)).
* [ONNX] Fix for expand -1 dim value (#34069) ([#34069](https://github.com/pytorch/pytorch/pull/34069)).
* ONNX Export Support for CrossEntropyLoss (#33767) ([#33767](https://github.com/pytorch/pytorch/pull/33767)).
* Enable constant folding for Reshape (#31054) ([#31054](https://github.com/pytorch/pytorch/pull/31054)).
* Fix index put (#31552) ([#31552](https://github.com/pytorch/pytorch/pull/31552)).
* kill py2 onnx builds ([#31082](https://github.com/pytorch/pytorch/pull/31082)).
* disable onnx py3 gcc5 build (#31100) ([#31100](https://github.com/pytorch/pytorch/pull/31100)).
* Skip same tests in ONNX Python3 CI as in Python2 (#31827) ([#31827](https://github.com/pytorch/pytorch/pull/31827)).
* Added torchvision tests as part of ORT tests (#31835) ([#31835](https://github.com/pytorch/pytorch/pull/31835)).
* skip mask_rcnn test (#34734) ([#34734](https://github.com/pytorch/pytorch/pull/34734)).
