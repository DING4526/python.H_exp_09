from data_provider import *
from train1 import TCN_MLP,MODEL_RUN
import mindspore
from mindspore import nn,Tensor

#定义MULTI_STEP_MODEL_RUN类
mindspore.set_context(mode=mindspore.GRAPH_MODE) #设置为静态图模式
class MULTI_STEP_MODEL_RUN(MODEL_RUN): #定义MULTI_STEP_MODEL_RUN类
    def __init__(self, model, loss_fn, optimizer=None, grad_fn=None): #构造方法
        super().__init__(model, loss_fn, optimizer, grad_fn)
    def evaluate(self, dataset): #重定义evaluate方法
        self.model.set_train(False) #设置为测试模式
        ls_pred,ls_label=[],[] #分别用于保存预测结果和标签
        for data, label in dataset.create_tuple_iterator(): #遍历每批数据
            muti_step_pred = mindspore.numpy.zeros_like(label[:,:,PV_index])
            x = data
            for step in range(horizon):
                pred = self.model(x, step) #使用sa_tcn_mlp模型进行预测
                muti_step_pred[:,step:step+1,:] = pred[:,:,PV_index] #将当前时间步的预测结果保存到multi_step_pred中
                concat_op = mindspore.ops.Concat(axis=1)
                x = concat_op((x[:,1:,:], pred)) #将预测结果加到输入中
                x[:,-1:,OP_index] = label[:,step:step+1,OP_index] #OP控制变量无法预测、始终使用真实值
            ls_pred += list(muti_step_pred.asnumpy()) #保存预测结果
            ls_label += list(label[:,:,PV_index].asnumpy()) #保存标签
        return loss_fn(Tensor(ls_pred), Tensor(ls_label)),np.array(ls_pred),np.array(ls_label)


#用迭代多步预测方式重新训练TCN_MLP模型
multi_step_tcn_mlp = TCN_MLP() #创建TCN_MLP类对象multi_step_tcn_mlp
loss_fn = nn.MAELoss() #定义损失函数
multi_step_optimizer = nn.Adam(multi_step_tcn_mlp.trainable_params(), 1e-3) #使用Adam优化器
def multi_step_forward_fn(data, label): #定义多步预测前向计算的multi_step_forward_fn方法
    muti_step_pred = mindspore.numpy.zeros_like(label[:,:,PV_index+DV_index])
    x = data
    for step in range(horizon):
        pred = multi_step_tcn_mlp(x, step) #使用multi_step__tcn_mlp模型进行预测
        muti_step_pred[:,step:step+1,:] = pred[:,:,PV_index+DV_index] #将当前时间步的预测结果保存到multi_step_pred中
        concat_op = mindspore.ops.Concat(axis=1)
        x = concat_op((x[:,1:,:], pred)) #将预测结果加到输入中
        x[:,-1:,OP_index] = label[:,step:step+1,OP_index] #OP控制变量无法预测、始终使用真实值
    loss = loss_fn(muti_step_pred, label[:,:,PV_index+DV_index]) #根据损失函数计算PV和DV变量的损失值
    return loss, muti_step_pred #返回损失值和预测结果
multi_step_grad_fn = mindspore.value_and_grad(multi_step_forward_fn, None, multi_step_optimizer.parameters, has_aux=True) #获取用于计算梯度的函数
multi_step_model_run = MULTI_STEP_MODEL_RUN(multi_step_tcn_mlp, loss_fn, multi_step_optimizer, multi_step_grad_fn) #创建MULTI_STEP_MODEL_RUN类对象multi_step_model_run

if __name__ == '__main__':
    multi_step_model_run.train(train_dataset_t2, val_dataset_t2, 10, 'multi_step_tcn_mlp.ckpt') #调用multi_step_model_run.train方法完成训练