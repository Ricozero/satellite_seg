# CCF BCDI2017 卫星影像的AI分类与识别 线上Top1部分代码

## 运行环境
Python版本：3.7

主要依赖库：
* PyTorch
* torchvision
* [visdom](http://github.com/facebookresearch/visdom)
* [pydensecrf](http://github.com/lucasb-eyer/pydensecrf)

## 概述

* DenseNet121为基础网络，PSPNet作为分割的模型，多尺度训练/测试，CRF后处理等

* 训练数据集:

	* CCF初赛训练数据（s1）+ CCF复赛训练数据（s2）
	* CRF处理CCF训练数据（s1s2-crf）
	* Dstl训练数据

* 主要尝试的模型（原作者）:

	* 训练数据集s1s2 (pspnet-densenet-s1s2)
	* 训练数据集s1s2-crf2 (pspnet-densenet-s1s2-crf2)
	* 不同网络输入尺度 (pspnet-densenet-s1s2-320)
	* focal loss (pspnet-densenet-s1s2-crf2-fl)
	* 类别加权训练 (pspnet-densenet-s1s2-crf2-weight)

## 数据预处理

* [CCF数据集下载链接](https://pan.baidu.com/s/1nu8srUh)（提取码：al0x）（把`BDCI2017-seg`的内容放入`dataset/`）
	* 注：有Semi表示复赛，无Semi表示初赛
* [Dstl数据集下载链接](https://www.kaggle.com/c/dstl-satellite-imagery-feature-detection/data)（解压放入`dataset/dstl`）

执行预处理：

	./run_preparation.sh


## 训练
`run_train.sh` 根据Overview里面的模型设置，更改`train_dir`选择对应的训练数据和`model_name`设置训练的模型
* pspnet-densenet-s1s2-320,更改--image_rows 和 --img_cols 为320
* 在`run_train.sh`，除了pspnet-densenet-s1s2-crf2-fl调用`train-fl.py`, 其它模型用`train.py`
* 对于pspnet-densenet-s1s2-crf2-weight,更改`train.py`中的`weights_per_class` 为[0,1,1,3,3]，默认[0,1,1,1,1]

## 测试 & Vote

* `run_test.sh`：更改`model_name`选择对应的模型测试
* `run_vote.sh`：更改`model_name`,对同一模型的不同epoch测试结果进行投票，得到该单模型结果
* `submit.sh`：每个模型的测试目录 use_crf（e.g. ./submit.sh results/pspnet-densenet-s1s2-crf2/vote 1）
