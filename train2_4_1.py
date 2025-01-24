from data_provider import *
from train2_2 import MULTI_STEP_MODEL_RUN
import mindspore
from mindspore import nn,Tensor


mindspore.set_context(mode=mindspore.GRAPH_MODE) #设置为静态图模式
class TCN_MLP_with_Bias_Block(nn.Cell): #定义TCN_MLP_with_Bias_Block类
    def __init__(self): #构造方法
        super().__init__() #调用父类的构造方法
        #比MULTI_STEP_TCN_MLP类增加一个偏差块
        self.bias_block = nn.SequentialCell(
            nn.Dense(sensor_num, 64),
            nn.ReLU(),
            nn.Dense(64, 32),
            nn.ReLU(),
            nn.Dense(32, sensor_num),
            nn.ReLU(),
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(32,1), pad_mode='valid')
        )
        #对预测时间步数据的嵌入编码操作
        self.step_embedding = nn.Embedding(horizon, sensor_num)
        #对不同传感器的数据做融合（提取传感器数据间的关联特征）
        self.spatial_mlp = nn.SequentialCell(
            nn.Dense(sensor_num, 128),
            nn.ReLU(),
            nn.Dense(128, 64),
            nn.ReLU(),
            nn.Dense(64, 32),
            nn.ReLU(),
            nn.Dense(32, sensor_num)
        )
        #对时间序列做卷积（提取时间点数据间的关联特征）
        self.tcn = nn.SequentialCell(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3,1), pad_mode='valid'),
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3,1), pad_mode='valid'),
        )
        #通过一个卷积层得到最后的预测结果
        self.final_conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(26, 1), pad_mode='valid') #使用26*1卷积核，不补边
    def construct(self, x, iter_step): #construct方法
        h = self.spatial_mlp(x) #经过spatial_mlp空间处理后，得到的数据h的形状：[batch_size, 30, 23]
        #输入数据x的形状：[batch_size, 30, 23]
        h = x + h #残差连接，将x和h对应元素相加，得到的数据h的形状：[batch_size, 30, 23]
        h = h.unsqueeze(1) #根据卷积操作需要，将3维数据升为4维数据：[batch_size, 1, 30, 23]
        h = self.tcn(h) #经过tcn时间卷积后，得到的数据x的形状：[batch_size, 1, 26, 23]
        y = self.final_conv(h) #通过26*1的卷积操作后，得到的数据y的形状：[batch_size, 1, 1, 23]
        y = y.squeeze(1) #将前面增加的维度去掉，得到的数据y的形状：[batch_size, 1, 23]
        #计算时间步数据的嵌入编码
        iter_step_tensor = mindspore.numpy.full((x.shape[0], 1), iter_step, dtype=mindspore.int32)
        step_embedding = self.step_embedding(iter_step_tensor) #step_embedding的形状：[batch_size,1,23]
        #将输入数据x与时间步数据的嵌入编码拼接
        concat_op = mindspore.ops.Concat(axis=1)
        bias_input = concat_op((x, step_embedding,y)) #bias_input的形状：[batch_size,32,23]
        bias_input = bias_input.unsqueeze(1)  # 根据卷积操作需要，将3维数据升为4维数据：[batch_size, 1, 32, 23]
        bias_output = self.bias_block(bias_input) #[batch_size, 1, 1, 23]
        bias_output = bias_output.squeeze(1) # [batch_size, 1, 23]
        #加上偏差块的预测结果
        y = y + bias_output #y的形状：[batch_size, 1, 23]
        return y #返回计算结果


tcn_mlp_bias = TCN_MLP_with_Bias_Block() #创建TCN_MLP_with_Bias_Block类对象tcn_mlp_bias
loss_fn = nn.MAELoss() #定义损失函数
multi_step_optimizer = nn.Adam(tcn_mlp_bias.trainable_params(), 1e-3) #使用Adam优化器
def multi_step_forward_fn(data, label): #定义多步预测前向计算的multi_step_forward_fn方法
    muti_step_pred = mindspore.numpy.zeros_like(label[:,:,PV_index+DV_index])
    x = data
    for step in range(horizon):
        pred = tcn_mlp_bias(x, step) #使用tcn_mlp_bias模型进行预测
        muti_step_pred[:,step:step+1,:] = pred[:,:,PV_index+DV_index] #将当前时间步的预测结果保存到multi_step_pred中
        concat_op = mindspore.ops.Concat(axis=1)
        x = concat_op((x[:,1:,:], pred)) #将预测结果加到输入中
        x[:,-1:,OP_index] = label[:,step:step+1,OP_index] #OP控制变量无法预测、始终使用真实值
    loss = loss_fn(muti_step_pred, label[:,:,PV_index+DV_index]) #根据损失函数计算PV和DV变量的损失值
    return loss, muti_step_pred #返回损失值和预测结果
multi_step_grad_fn = mindspore.value_and_grad(multi_step_forward_fn, None, multi_step_optimizer.parameters, has_aux=True) #获取用于计算梯度的函数
multi_step_model_run = MULTI_STEP_MODEL_RUN(tcn_mlp_bias, loss_fn, multi_step_optimizer, multi_step_grad_fn) #创建MODEL_RUN类对象model_run

if __name__ == '__main__':
    multi_step_model_run.train(train_dataset_t2, val_dataset_t2, 10, 'tcn_mlp_bias.ckpt') #调用model_run.train方法完成训练