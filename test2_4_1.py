from train2_4_1 import *
from mindspore import nn
import matplotlib.pyplot as plt

tcn_mlp_bias = TCN_MLP_with_Bias_Block() #创建TCN_MLP_with_Bias_Block类对象tcn_mlp_bias
loss_fn = nn.MAELoss() #定义损失函数
multi_step_model_run = MULTI_STEP_MODEL_RUN(tcn_mlp_bias, loss_fn) #创建MULTI_STEP_MODEL_RUN类对象multi_step_model_run
print("开始测试......")
train_loss,_,_ = multi_step_model_run.test(train_dataset_t2, 'tcn_mlp_bias.ckpt') #计算训练集损失
val_loss,_,_ = multi_step_model_run.test(val_dataset_t2, 'tcn_mlp_bias.ckpt') #计算验证集损失
test_loss,preds,labels = multi_step_model_run.test(test_dataset_t2, 'tcn_mlp_bias.ckpt') #计算测试集损失
print('训练集损失：{0}，验证集损失：{1}，测试集损失：{2}'.format(train_loss,val_loss,test_loss))
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False
_,axes = plt.subplots(5,1,figsize=(8, 16))
interval = int(horizon/5)
for step in range(5):
    axes[step].set_title('第%d个时间步的预测结果'%(step*interval+1))
    axes[step].plot(range(1,101), preds[:100,step*interval,0], color='Red') # 绘制第1个传感器的前100条数据的预测结果
    axes[step].plot(range(1,101), labels[:100,step*interval,0], color='Blue') # 绘制第1个传感器的前100条数据的标签
plt.show()
print("测试完毕！")
