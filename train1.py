from data_provider import *
from mindspore import nn,Tensor #导入Tensor类
import mindspore #导入mindspore
import numpy as np #导入numpy工具包


#定义TCN_MLP类
class TCN_MLP(nn.Cell): #定义TCN_MLP类
    def __init__(self): #构造方法
        super().__init__() #调用父类的构造方法
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
    def construct(self, x, step=None): #construct方法
        #输入数据x的形状：[batch_size, 30, 23]
        h = self.spatial_mlp(x) #经过spatial_mlp空间处理后，得到的数据h的形状：[batch_size, 30, 23]
        x = x + h #残差连接，将x和h对应元素相加，得到的数据x的形状：[batch_size, 30, 23]
        x = x.unsqueeze(1) #根据卷积操作需要，将3维数据升为4维数据：[batch_size, 1, 30, 23]
        x = self.tcn(x) #经过tcn时间卷积后，得到的数据x的形状：[batch_size, 1, 26, 23]
        y = self.final_conv(x) #通过26*1的卷积操作后，得到的数据y的形状：[batch_size, 1, 1, 23]
        y = y.squeeze(1) #将前面增加的维度去掉，得到的数据y的形状：[batch_size, 1, 23]
        return y #返回计算结果


#定义MODEL_RUN类
mindspore.set_context(mode=mindspore.GRAPH_MODE) #设置为静态图模式
class MODEL_RUN: #定义MODEL_RUN类
    def __init__(self, model, loss_fn, optimizer=None, grad_fn=None): #构造方法
        self.model = model #设置模型
        self.loss_fn = loss_fn #设置损失函数
        self.optimizer = optimizer #设置优化器
        self.grad_fn = grad_fn #设置梯度计算函数
    def _train_one_step(self, data, label): #定义用于单步训练的_train_one_step方法
        (loss, _), grads = self.grad_fn(data, label) #根据数据和标签计算损失和梯度
        self.optimizer(grads) #根据梯度进行模型优化
        return loss #返回损失值
    def _train_one_epoch(self, train_dataset): #定义用于一轮训练的_train_one_epoch方法
        self.model.set_train(True) #设置为训练模式
        for data, label in train_dataset.create_tuple_iterator(): #取出每一批数据
            self._train_one_step(data, label) #调用_train_one_step方法进行模型参数优化
    def evaluate(self, dataset, step=None): #定义用于评估模型的evaluate方法
        self.model.set_train(False) #设置为测试模式
        ls_pred,ls_label=[],[] #分别用于保存预测结果和标签
        for data, label in dataset.create_tuple_iterator(): #遍历每批数据
            pred = self.model(data) #使用模型对一批数据进行预测
            ls_pred += list(pred[:,:,PV_index].asnumpy()) #保存预测结果
            ls_label += list(label[:,:,PV_index].asnumpy()) #保存标签
        return loss_fn(Tensor(ls_pred), Tensor(ls_label)), np.array(ls_pred), np.array(ls_label)
    def train(self, train_dataset, val_dataset, max_epoch_num, ckpt_file_path): #定义用于训练模型的train方法
        min_loss = Tensor(np.finfo(np.float32).max) #将min_loss设置为最大值
        print('开始训练......')
        for epoch in range(1,max_epoch_num+1): #迭代训练
            print('第{0}/{1}轮'.format(epoch,max_epoch_num)) #输出当前迭代轮数/总轮数
            self._train_one_epoch(train_dataset) #调用_train_one_epoch完成一轮训练
            train_loss,_,_ = self.evaluate(train_dataset) #在训练集上计算模型损失值
            eval_loss,_,_ = self.evaluate(val_dataset) #在验证集上计算模型损失值
            print('训练集损失：{0}，验证集损失：{1}'.format(train_loss,eval_loss))
            if eval_loss < min_loss: #如果验证集损失值低于原来保存的最小损失值
                mindspore.save_checkpoint(self.model, ckpt_file_path) #更新最优模型文件
                min_loss = eval_loss #保存新的最小损失值
        print('训练完成！')
    def test(self, test_dataset, ckpt_file_path): #定义用于测试模型的test方法
        mindspore.load_checkpoint(ckpt_file_path, net=self.model) #从文件中加载模型
        loss,preds,labels = self.evaluate(test_dataset) #在测试集上计算模型损失值
        return loss,preds,labels #返回损失值


# 进行单步预测模型的训练
tcn_mlp = TCN_MLP()  # 创建TCN_MLP类对象tcn_mlp
loss_fn = nn.MAELoss()  # 定义损失函数
optimizer = nn.Adam(tcn_mlp.trainable_params(), 1e-3)  # 使用Adam优化器
def forward_fn(data, label):  # 定义前向计算的forward_fn函数
    pred = tcn_mlp(data)  # 使用tcn_mlp模型进行预测
    loss = loss_fn(pred[:, :, PV_index], label[:, :, PV_index])  # 根据损失函数计算PV变量的损失值
    return loss, pred  # 返回损失值和预测结果
grad_fn = mindspore.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)  # 获取用于计算梯度的函数
model_run = MODEL_RUN(tcn_mlp, loss_fn, optimizer, grad_fn)  # 创建MODEL_RUN类对象model_run

if __name__ == '__main__':
    model_run.train(train_dataset=train_dataset_t1, val_dataset=val_dataset_t1, max_epoch_num=20,
                    ckpt_file_path='tcn_mlp.ckpt')  # 调用model_run.train方法完成训练




