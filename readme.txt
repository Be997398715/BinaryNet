本次实现Binary_Net网络，为二值化神经网络的典型代表，将权值和激活函数都约束在[-1，+1]，不使用传统的float32计算，使用int64，大大
降低了计算成本和网络参数，而且也能达到较好的效果，为嵌入式设备的网络搭建提供了很大的帮助。

相关资料可以查看：https://blog.csdn.net/sinat_24143931/article/details/78635198（源码实现）
                  https://www.colabug.com/3912698.html
                  https://blog.csdn.net/nature553863/article/details/80653521（源码理解）
                  
训练： python train.py(训练时需要全精度，将值约束在[-1,+1])
测试： python test.py(将权重文件修改并二值化，实现模型压缩)
测试： python predict_and_visuable.py(TTA和特征可视化)
模型相关：models文件夹下
测试图片：test_images文件夹下
保存模型权重及可视化图片：logs文件夹下

源码包含了train和inference的代码，我这里只有train的，最终我进行了权重的修改进行了二值化，且精度一样不低，
BinaryNet 的最大优点是可以 XNOR-计数 运算替代复杂的乘法-加法操作，之后只要在移动端部署即可(参考原作者的inference代码)
但这部分也是较为复杂的，我暂时还没有时间做。

相关模型链接：链接：https://pan.baidu.com/s/1B2XArrH-2peZWelxwr6ldQ 提取码：ji3f 

########################################### 复现成功 ###########################################################
