## 第三届中国AI+创新创业大赛：半监督学习目标定位竞赛第27名方案

## 项目描述
**比赛背景**
第三届中国AI+创新创业大赛由中国人工智能学会主办，半监督学习目标定位竞赛分赛道要求选手基于少量有标注数据训练模型，使分类网络具有目标定位能力，实现半监督目标定位任务。

中国人工智能学会（Chinese Association for Artificial Intelligence，CAAI）成立于1981年，是经国家民政部正式注册的我国智能科学技术领域唯一的国家级学会，是全国性4A级社会组织，挂靠单位为北京邮电大学；是中国科学技术协会的正式团体会员，具有推荐“两院院士”的资格。

中国人工智能学会目前拥有51个分支机构，包括43个专业委员会和8个工作委员会，覆盖了智能科学与技术领域。学会活动的学术领域是智能科学技术，活动地域是中华人民共和国全境，基本任务是团结全国智能科学技术工作者和积极分子通过学术研究、国内外学术交流、科学普及、学术教育、科技会展、学术出版、人才推荐、学术评价、学术咨询、技术评审与奖励等活动促进我国智能科学技术的发展，为国家的经济发展、社会进步、文明提升、安全保障提供智能化的科学技术服务。

中国“AI+”创新创业大赛由中国人工智能学会发起主办，是为了配合实施创新驱动助力工程，深入开展服务企业技术创新活动，进一步提高我国文化建设和实践创新能力，展示智能科学与技术等相关学科建设的新经验、新成果，促进专业内涵的建设而发起的综合性大赛平台。

飞桨PaddlePaddle作为中国首个自主研发、功能完备、开源开放的产业级深度学习平台，为本次比赛的参赛选手提供了集深度学习核心训练和推理框架、基础模型库、端到端开发套件和丰富的工具组件于一体的一站式服务。百度大脑AI Studio作为官方指定且唯一的竞赛日常训练平台，为参赛选手提供高效的学习和开发环境，更有亿元Tesla V100算力免费赠送，助力选手取得优异成绩。
**赛题背景**
半监督学习(Semi-Supervised Learning)是指通过大量无标记数据和少量有标记数据完成模型训练，解决具有挑战性的模式识别任务。近几年，随着计算硬件性能的提升和大量大规模标注数据集的开源，基于深度卷积神经网络(Deep Convolutional Neural Networks, DCNNs)的监督学习研究取得了革命性进步。然而，监督学习模型的优异性能要以大量标注数据作为支撑，可现实中获得数量可观的标注数据十分耗费人力物力(例如，获取像素级标注数据)。于是， 半监督学习逐渐成为深度学习领域的热门研究方向，只需要少量标记数据就可以完成模型训练过程，更适用于现实场景中的各种任务。

**比赛任务**
本次比赛要求选手基于少量有标签的数据训练模型，使分类网络具有目标定位能力，实现半监督目标定位任务。每- -位参赛选手仅可以使用ImageNet大型视觉识别竞赛(LSVRC)的训练集图像作为训练数据，其中有标签的训练数据仅可以使用大赛组委会提供的像素级标注。

**数据集介绍**
训练数据集包括50,000幅像素级标注的图像，共包含500个类，每个类100幅图像;
A榜测试数据集包括1 1,878幅无标注的图像;
B榜测试数据集包括10,989幅无标注的图像。
评价指标
本次比赛使用loU曲线作为评价指标，即利用预测的目标的定位概率图，计算不同阈值下预测结果与真实目标之间的IoU分数，最后取一个最高点作为最终的分数。在理想状态下，loU曲线最高值接近1.0,对应的阈值为255,因为阈值越高，目标对象与背景的对比度越高。


## 模型构建
```
主要用到paddle的paddleseg代码进行更改

1.更改了损失函数，去掉了paddleseg\models\losses\cross_entropy_loss.py中的coef计算
2.在dataset中更改了标签数，加入了class_idx
3.在transforms中把输入的lables_dims[axis]更改为1

在上传的paddleseg.zip中，加入data和mdel可直接运行
model可以从下面链接提取  链接：https://pan.baidu.com/s/17lOWJbHN-FtxFGNhwCwpAw   提取码：61t0

In [ ]:
#解压一下略小改之后的PaddleSeg，解压一次就可以注释掉了

!unzip -oq /home/aistudio/PaddleSeg.zip

In [ ]:
#解压数据集至data/目录

!unzip -qo data/data95249/train_50k_mask.zip -d data/

!unzip -oq data/data100087/B榜测试数据集.zip -d data/

!unzip -oq data/data95249/train_image.zip -d data/

数据集划分
执行一次就行了，之后可直接跳到后面的参数配置及训练


import sys
sys.path.append("PaddleSeg")
import paddleseg
import paddle
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import random

#设置随机数种子

random.seed(2021)


def write_txt(file_name, imgs_path, labels_path=None, mode='train', val_pro=0.2):
    assert mode=="train" or mode=="test", "ERROR:mode must be train or test."
    if mode!="test":
        train_path = []
        for idx, f_path in enumerate(imgs_path):
            for i_path in sorted(os.listdir(f_path)):
                path1 = os.path.join(f_path, i_path) 
                path2 = os.path.join(labels_path[idx], i_path)
                train_path.append((path1, path2, str(idx)))

        if val_pro>=0 and val_pro<=1:
            #打乱数据
            random.shuffle(train_path)
            val_len = int(len(train_path)*val_pro)
            val_path = train_path[:val_len]
            train_path = train_path[val_len:]
            with open(file_name[0], 'w') as f:
                for path in train_path:
                    f.write(path[0]+" "+path[1]+" "+path[2]+"\n")
            with open(file_name[1], 'w') as f:
                for path in val_path:
                    f.write(path[0]+" "+path[1]+" "+path[2]+"\n")  
            return len(train_path), val_len
        else:
            with open(file_name[0], 'w') as f:
                for path in train_path:
                    f.write(path[0]+" "+path[1]+" "+path[2]+"\n") 
            return len(train_path), 0
    else:
        with open(file_name, 'w') as f:
            for path in imgs_path:
                img_path = os.path.join(test_path, path)
                f.write(img_path+"\n")

def create_txt(data_root, train_imgs_dir=None, train_labels_dir=None, test_dir=None, val_pro=0.2):
    if train_imgs_dir is not None:
        if os.path.exists("train.txt"):
            os.remove("train.txt")
        if os.path.exists("val.txt"):
            os.remove("val.txt")
        train_imgs_dir = os.path.join(data_root, train_imgs_dir)
        train_labels_dir = os.path.join(data_root, train_labels_dir)
        file_names = os.listdir(train_imgs_dir)
        file_names = sorted(file_names)
        train_imgs_path, train_labels_path =[], []
        for na in file_names:
            train_imgs_path.append(os.path.join(train_imgs_dir, na))
            train_labels_path.append(os.path.join(train_labels_dir, na))
        train_len, val_len = write_txt(["train.txt", "val.txt"], train_imgs_path, train_labels_path, mode='train', val_pro=val_pro)

        print("训练数据整理完毕！训练集长度：{}，验证集长度：{}， 类别数：{}".format(train_len, val_len, len(file_names)))

    if test_dir is not None:
        if os.path.exists("test.txt"):
            os.remove("test.txt")
        global test_path
        test_path = os.path.join(data_root, test_dir)
        test_imgs_path_list = sorted(os.listdir(test_path))
        write_txt("test.txt", test_imgs_path_list, mode="test")
        print("测试数据整理完毕！测试集长度：{}".format(len(test_imgs_path_list)))

data_root = "data"
train_imgs_dir = "train_image"
train_labels_dir = "train_50k_mask"
test_dir = "test_image"
create_txt(data_root, train_imgs_dir, train_labels_dir, test_dir, val_pro=0.2)
参数配置及训练
在my_deeplabv3.yml中修改参数配置，利用deeplabv3P参数进行训练预测，更改了dataset的相对应参数，不然和训练标签不匹配。重启后直接运行下一行代码训练及验证，忘记保留checkpoint


!python PaddleSeg/train.py --config my_deeplabv3.yml --do_eval --use_vdl --save_dir /home/aistudio/output_deeplabv3_1 --save_interval 2000
推理
已在PaddleSeg中的\paddleseg\core\predict.py做了修改,增加了单通道label图像，保存在results路径，可以直接预测出结果


!python PaddleSeg/predict.py --config my_deeplabv3.yml --model_path output_deeplabv3_1/best_model/model.pdparams --image_path data/test_image --save_dir output/result_1 #--aug_pred --flip_horizontal --flip_vertical
压缩结果，提交文件
第三届中国AI+创新创业大赛：半监督学习目标定位竞赛


%cd output/result_1/results
!zip -r -oq /home/aistudio/pred.zip ./
%cd /home/aistudio
改进方向
试试其他模型架构，或进一步自己改进
试用一下其他数据增强操作
调参
等等
```
## 使用方式
A：在AI Studio上[运行本项目](https://aistudio.baidu.com/aistudio/projectdetail/2247720)
B：此处由项目作者进行撰写使用方式。
