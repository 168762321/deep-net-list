#训练自己的分类器
import torch
import torchvision.transforms as transforms
from PIL import Image
from lenet import LeNet


def main():
    transform = transforms.Compose([transforms.Resize((32,32)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

    # 导入自己的数据
    im = Image.open('D:\\python\\lenet\\6.jpg')  #记得换测试图像路径
    im = transform(im)  # [C, H, W]
    im = torch.unsqueeze(im, dim=0)  # 对数据增加一个新维度，因为tensor的参数是[batch, channel, height, width] 

    # 实例化网络，加载训练好的模型参数
    net = LeNet()
    net.load_state_dict(torch.load('D:\\python\\lenet\\Lenet.pth')) #记得换训练参数路径

    # 预测
    classes = ('飞机', '车', '鸟', '猫', '鹿', '狗', '青蛙', '房子', '船', '卡车')
    with torch.no_grad():
        outputs = net(im)
        predict = torch.max(outputs, dim=1)[1].data.numpy()
    print('图像是：',classes[int(predict)])

if __name__=="__main__":
    main()