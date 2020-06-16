我们希望使用机械臂的视觉来为场景建模，其中必要的需求是检测图像中的物体，并估计它们的6D pose。

我们使用YCB数据集来制作demo。

选择使用DenseFusion进行6D pose估计。

因为DenseFusion需要物体detection和segmentation作为前置条件，因此需要在YCB上训练mask RCNN模型。

在这里，我们使用名为mmdetection的库，从而方便地完成mask RCNN的训练。

使用mmdetection时，需要将ycb数据集的格式调整到与coco一样的格式，因此需要使用cocoAPI进行数据格式转换。

下面就按照时间顺序来叙述编写代码的过程。

### YCB -> coco 数据集格式转换

#### 图片搬运
coco格式的数据集下一般是由images和annotations两个子文件夹组成的。

运行imageSetPorter，需要指定的参数有：已有的YCB_Video_Dataset的位置，还有输出的Custom_YCB_Video_Dataset的保存路径，注意，这些路径都是相对于python工作路径的相对路径。

那我们这里的默认路径是，YCB_Video_Dataset被放在.../Robotics/perception/datasets/YCB_Video_Dataset下，而被输出的Custom_YCB_Video_Dataset和YCB_Video_Dataset在同样的文件夹下。

首先，在perception文件夹下创建datasets文件夹，将YCB_Video_Dataset软连接到默认路径，然后在datasets文件夹下创建Custom_Video_Dataset文件夹，并创建annotations和images子文件夹。

在perception工作路径下运行python imageSetPorter.py，它会在YCB_Video_Dataset中，根据image_sets下的trainval.txt，将data文件夹下的rgb图片搬运到输出数据集的images文件夹下。

#### annotation 搬运
annoCreator.py将YCB数据集的annotation转换为coco格式，保存到指定的自定义数据集路径下，请使得annoCreator.py与imageSetPorter.py的命令行参数相同，它们默认是相同的。

### 使用mmdetection训练coco格式的YCB数据集
mmdetection是一个集成了很多目标检测、分割、关键点检测算法的包。它截止到2020年中旬，release了1.*和2.*两种版本，其中2.0之后的版本与1.*的发布是互不兼容的。

mmdetection 1.*版本适用于pyTorch 1.1-1.4版本，而2.0版本适用于pyTorch 1.3+版本，安装前要仔细确认docs/install.md中不同release适用的cuda与pyTorch版本。

我在cuda version为9.0的机器上踩坑，因此本教程采用了mmdetection-1.0.0版本，请依照mmdetection的install.md进行安装。

然后按照mmdetection的教程使用先前自定义的数据集，以下是我按照mmdtection教程所做的操作。

首先需要编写自定义数据集的dataset类，在mmdet/datasets目录下创建ycb_dataset.py，写了YcbDataset类，它继承了CocoDataset，并将它添加到mmdet/datasets/__init__.py当中。

关键在于为我们的训练任务编写一个config文件，本教程以configs/mask_rcnn_r50_fpn_1x.py为backbone，将它修改为configs/ycb_mask_rcnn_r50_fpn_1x.py，注意要将config中的data_root变量调整为Custom_YCB_Video_Dataset的路径位置，其中默认工作路径在Robotics/perception。

最终，我在perception文件夹下创建launch_ycb_train.sh和launch_ycb_test.sh，在launch_ycb_train.sh中可以制定gpu数量。其他使用细节请参照mmdetection的教程。