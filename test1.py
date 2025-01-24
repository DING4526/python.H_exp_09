from mindspore import nn
from train1 import *
import matplotlib.pyplot as plt

#进行单步预测模型的测试
tcn_mlp = TCN_MLP() #创建TCN_MLP类对象tcn_mlp
loss_fn = nn.MAELoss() #定义损失函数
model_run = MODEL_RUN(tcn_mlp, loss_fn) #创建MODEL_RUN类对象model_run
print("开始测试......")
train_loss,_,_ = model_run.test(train_dataset_t1, 'tcn_mlp.ckpt') #计算训练集损失
val_loss,_,_ = model_run.test(val_dataset_t1, 'tcn_mlp.ckpt') #计算验证集损失
test_loss,preds,labels = model_run.test(test_dataset_t1, 'tcn_mlp.ckpt') #计算测试集损失
print('训练集损失：{0}，验证集损失：{1}，测试集损失：{2}'.format(train_loss,val_loss,test_loss))
plt.figure(figsize=(8, 4))
plt.plot(range(1,101), preds[:100,0,0], color='Red') # 绘制第1个传感器的前100条数据的预测结果
plt.plot(range(1,101), labels[:100,0,0], color='Blue') # 绘制第1个传感器的前100条数据的标签
plt.show()
print("测试完毕！")
