#处理数据
import torch
from torch.functional import Tensor
from torch.nn.functional import batch_norm
import torchvision
import torch.nn as nn
from lenet import LeNet
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import time
from torch import utils


#输入图像预处理 shape=长X宽x通道数
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
# 导入训练数据
train_set = torchvision.datasets.CIFAR10(root='./data',
                                        train=True,       # 表示是数据集中的训练集
                                        download=False,    # 第一次运行时为True，下载数据集，下载完成后改为False
                                        transform=transform)

train_loader = torch.utils.data.DataLoader(train_set, 	  # 导入的训练集
										batch_size=50,    # 每批训练的样本数
                                        shuffle=False,    # 是否打乱训练集
                                        num_workers=0)

# 导入10000张测试图片
test_set = torchvision.datasets.CIFAR10(root='./data', 
										train=False,	# 表示是数据集中的测试集
                                        download=False,transform=transform)
# 加载测试集
test_loader = torch.utils.data.DataLoader(test_set, 
										  batch_size=10000, # 每批用于验证的样本数
										  shuffle=False, num_workers=0)
# 获取测试集中的图像和标签，用于accuracy精度计算
test_data_iter = iter(test_loader)
test_image, test_label = test_data_iter.next()

net = LeNet()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net.to(device)
loss_f = nn.CrossEntropyLoss()                          #定义损失函数，交叉熵函数
optimizer = optim.Adam(net.parameters(),lr = 0.001)     #优化器

for epoch in range(20):  #一个epoch就是一个训练周期
    running_loss = 0.0
    time_start = time.perf_counter()

    for step, data in enumerate(train_loader, start=0):   # 遍历训练集，step从0开始计算
        inputs, labels = data                             # 获取训练集的图像和标签
        optimizer.zero_grad()                             # 清除历史梯度
        
        # forward + backward + optimize 正向传播+反向传播+优化参数
        outputs = net(inputs.to(device))  	             # 正向传播
        loss = loss_f(outputs, labels.to(device))        # 计算损失
        loss.backward() 					  # 反向传播
        optimizer.step() 					  # 优化器更新参数

        running_loss+=loss.item()
        if step %1000==999:
            with torch.no_grad(): #在以下步骤中（验证过程中）不用计算每个节点的损失梯度，防止内存占用
                outputs = net(test_image.to(device))
                predict_y = torch.max(outputs,dim=1)[1] # 以output中值最大位置对应的索引（标签）作为预测输出
                accuracy = (predict_y == test_label.to(device)).sum().item() / test_label.size(0)
                print("在gpu训练还是cpu：",device)
                print('[%d, %5d] 训练误差train_loss: %.3f  测试精度test_accuracy: %.3f' % 
                      (epoch + 1, step + 1, running_loss / 500, accuracy)) # 打印epoch，step，loss，accuracy
                
                print('耗时 %f s' % (time.perf_counter() - time_start))        # 打印耗时
                running_loss = 0.0

print('结束训练')

# 保存训练得到的参数
save_path = 'D:\\python\\lenet\\lenet.py'  #记得改成自己的地址
torch.save(net.state_dict(), save_path)





