第三届中国AI+创新创业大赛：半监督学习目标定位竞赛第27名方案

## 项目描述
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

In [ ]:
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

In [ ]:
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
In [ ]:
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
In [ ]:
data_root = "data"
train_imgs_dir = "train_image"
train_labels_dir = "train_50k_mask"
test_dir = "test_image"
create_txt(data_root, train_imgs_dir, train_labels_dir, test_dir, val_pro=0.2)
参数配置及训练
在my_deeplabv3.yml中修改参数配置，利用deeplabv3P参数进行训练预测，更改了dataset的相对应参数，不然和训练标签不匹配。重启后直接运行下一行代码训练及验证，忘记保留checkpoint

In [ ]:
!python PaddleSeg/train.py --config my_deeplabv3.yml --do_eval --use_vdl --save_dir /home/aistudio/output_deeplabv3_1 --save_interval 2000
推理
已在PaddleSeg中的\paddleseg\core\predict.py做了修改,增加了单通道label图像，保存在results路径，可以直接预测出结果

In [13]:
#推理
!python PaddleSeg/predict.py --config my_deeplabv3.yml --model_path output_deeplabv3_1/best_model/model.pdparams --image_path data/test_image --save_dir output/result_1 #--aug_pred --flip_horizontal --flip_vertical
压缩结果，提交文件
第三届中国AI+创新创业大赛：半监督学习目标定位竞赛

In [ ]:
%cd output/result_1/results
!zip -r -oq /home/aistudio/pred.zip ./
%cd /home/aistudio
改进方向
试试其他模型架构，或进一步自己改进
试用一下其他数据增强操作
调参
等等

## 代码
```
在上传的paddleseg.zip中，加入data和mdel可直接运行
```
## 使用方式
A：在AI Studio上[运行本项目](https://aistudio.baidu.com/aistudio/usercenter)
B：此处由项目作者进行撰写使用方式。
