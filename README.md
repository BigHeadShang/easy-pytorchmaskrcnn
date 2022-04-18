# easy-pytorchmaskrcnn
## **you can train easily**
#
Mask-RCNN作为一个集检测、分割于一体的网络，被众多CVer一度追捧。目前在Github上也有不少开源的相关代码仓库，然而目前Github上开源的主要为TensorFlow版本的Mask RCNN以及基于Pytorch版本的高度集成网络架构Detectronv2。可以自己改动网络结构的Pytorch版本的源码少之又少，目前笔者能在Github上找到的基于Pytorch版本的MAskRCNN，仅仅是有一个写好的网络架构，训练文件仍需自己编写，这对深度学习的新手来说非常不友好，因此笔者仿照TensorFlow版，进行了代码迁移，仿写了一版基于Pytorch版的训练文件。以下为本版本的环境配置说明。总共两部分：第一部分为基于云服务器进行的环境配置，第二部分为基于Colab的环境配置
# 基础配置
``` CUDA 9.0
CUDNN 7.6.5
Python 3.6
Pytorch 0.4.1（预装）
Ubuntu 18.04
gcc 4.9
g++ 4.9
opencv-python
matplotlib
ipykernel
scikit_image
scipy==1.2.1
Pillow==6.0.0
```
# 云服务器环境配置
```!pip install torch-0.4.1-cp36-cp36m-linux_x86_64.whl
import os
os.chdir('/')
os.chdir('/etc/apt/')
!apt-get update
!apt-get install gcc-4.9
!apt-get install g++-4.9
!update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-4.9 20
!update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-4.9 20
os.chdir('/')
os.chdir('/mnt/pymaskrcnn/nms/src/cuda/')
!nvcc -c -o nms_kernel.cu.o nms_kernel.cu -x cu -Xcompiler -fPIC -arch=sm_60
os.chdir('/mnt/pymaskrcnn/nms/')
!python build.py
os.chdir('/mnt/pymaskrcnn/')
!pip install torch-0.3.0-cp36-cp36m-linux_x86_64.whl
os.chdir('/mnt/pymaskrcnn/roialign/roi_align/src/cuda/')
!nvcc -c -o crop_and_resize_kernel.cu.o crop_and_resize_kernel.cu -x cu -Xcompiler -fPIC -arch=sm_60
os.chdir('/mnt/pymaskrcnn/roialign/roi_align/')
!python build.py
os.chdir('/mnt/pymaskrcnn/')
```
# Colab
## 首先建立python环境
```
%%bash
MINICONDA_INSTALLER_SCRIPT=Miniconda3-4.5.4-Linux-x86_64.sh
MINICONDA_PREFIX=/usr/local
wget https://repo.continuum.io/miniconda/$MINICONDA_INSTALLER_SCRIPT
chmod +x $MINICONDA_INSTALLER_SCRIPT
./$MINICONDA_INSTALLER_SCRIPT -b -f -p $MINICONDA_PREFIX
import sys
_ = (sys.path.append("/usr/local/lib/python3.6/site-packages"))
```
## 然后将笔者的脚本clone，进行自动操作
```!git clone https://gist.github.com/c2e40071c94338ac162db232dbecd211.git
import os
os.chdir('/content/c2e40071c94338ac162db232dbecd211')
!bash keytoinstallcuda90andpytorch041.sh
# 将工作目录变动到网盘中
from google.colab import drive
drive.mount('/content/gdrive/')
!pip install http://download.pytorch.org/whl/cu90/torch-0.4.1-cp36-cp36m-linux_x86_64.whl
!pip install torchvision==0.2.1
import os
os.chdir('/content/gdrive/MyDrive/')
!sudo dpkg -i libcudnn7_7.5.0.56-1+cuda9.0_amd64.deb
!sudo dpkg -i libcudnn7-dev_7.5.0.56-1+cuda9.0_amd64.deb
os.chdir('/')
!sudo cp /usr/include/cudnn.h /usr/local/cuda/include
!sudo chmod a+x /usr/local/cuda/include/cudnn.h
!cat /usr/local/cuda/include/cudnn.h | grep CUDNN_MAJOR -A 2
!cp sources.list sources.list.bk 
os.chdir('/')
os.chdir('/etc/apt/')
!apt-get update
!apt-get install gcc-4.9
!apt-get install g++-4.9
!update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-4.9 20
!update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-4.9 20
os.chdir('/')
os.chdir('/content/gdrive/MyDrive/pytorch-mask-rcnn-master/nms/src/cuda/')
!nvcc -c -o nms_kernel.cu.o nms_kernel.cu -x cu -Xcompiler -fPIC -arch=sm_60
os.chdir('/content/gdrive/MyDrive/pytorch-mask-rcnn-master/nms/')
!python build.py
os.chdir('/content/gdrive/MyDrive/pytorch-mask-rcnn-master/')
!pip install torch-0.3.0-cp36-cp36m-linux_x86_64.whl
os.chdir('/content/gdrive/MyDrive/pytorch-mask-rcnn-master/roialign/roi_align/src/cuda/')
!nvcc -c -o crop_and_resize_kernel.cu.o crop_and_resize_kernel.cu -x cu -Xcompiler -fPIC -arch=sm_60
os.chdir('/content/gdrive/MyDrive/pytorch-mask-rcnn-master/roialign/roi_align/')
!python build.py
os.chdir('/content/gdrive/MyDrive/pytorch-mask-rcnn-master/')
!/usr/local/bin/python3.6 -m pip install -U pip
!/usr/local/bin/pip3 install opencv-python matplotlib ipykernel scikit_image scipy==1.2.1 pillow==6.0.0
!python train_test.py
```


