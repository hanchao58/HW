import numpy as np
import torch
from torch import tensor

# data = [[1, 2], [3, 4]]
# x_data = torch.tensor(data)  # 列表转换为tensor
# b = tensor([1., 2., 3., ])  # 浮点型列表
# c = torch.ones_like(b)  # 类似的有zeros_like 和rand_like，列表小括号，元祖中括号
# a = torch.rand((2, 2))  # 可以查询的属性有dtype，shape，device
# print(type(c))
# print(a.dtype)
# # Tensor的操作：算术、线性代数、采样，可以在CPU与GPU上执行
# if torch.cuda.is_available():
#     a = a.to('cuda')
# torch.is_tensor(a)
#
# print(torch.gather(torch.tensor([[1, 2], [3, 4]]), 1, torch.tensor([[1, 1], [0, 1]])))
#         #out[0][0] = input[0][ index[0][0] ] = input[0][1] = 2
#
#         #out[0][1] = input[0][ index[0][1] ] = input[0][1] = 2
#
#         #out[1][0] = input[1][ index[1][0] ] = input[1][0] = 3
#
#         #out[1][1] = input[1][ index[1][1] ] = input[1][1] = 4
# Numerical Operations
import math
import numpy as np

# Reading/Writing Data
import pandas as pd
import os
import csv

# For Progress Bar
from tqdm import tqdm

# Pytorch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

# For plotting learning curve
from torch.utils.tensorboard import SummaryWriter


def same_seed(seed):
    """Fixes random number generator seeds for reproducibility."""  # 使用种子以便随机复现
    torch.backends.cudnn.deterministic = True  # 每次返回的卷积算法是固定的，使得每次运行结果一样
    torch.backends.cudnn.benchmark = False  # 因为网络结构经常变，Ture表示每次 PyTorch 都会自动来根据新的卷积场景做优化
    np.random.seed(seed)  # 用于产生随机数
    torch.manual_seed(seed)  # 设置CPU生成随机数的种子，方便下次复现实验结果
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # 设置GPU


def train_valid_split(data_set, valid_ratio, seed):
    """Split provided training data into training set and validation set"""
    valid_set_size = int(valid_ratio * len(data_set))
    train_set_size = len(data_set) - valid_set_size
    train_set, valid_set = random_split(data_set, [train_set_size, valid_set_size],
                                        generator=torch.Generator().manual_seed(seed))
    # 用于分割数据集，generator用于随机排列的生成器，torch.Generator().manual_seed与torch.manual_seed一样
    return np.array(train_set), np.array(valid_set)  # array方法返回数组对象或任何序列


def predict(test_loader, model, device):
    model.eval()  # Set your model to evaluation mode.
    # 不启用 BatchNormalization 和 Dropout，保证BN和dropout不发生变化，pytorch框架会自动把BN和Dropout固定住，不会取平均，而是用训练好的值，不然的话，一旦test
    # 的batch_size过小，很容易就会被BN层影响结果。
    preds = []  # 初始化为list
    for x in tqdm(test_loader):  # Tqdm 是一个快速，可扩展的Python进度条，可以在 Python 长循环中添加一个进度提示信息
        x = x.to(device)
        with torch.no_grad():  # 在该模块下，所有计算得出的tensor的requires_grad都自动设置为False，反向传播时就不会自动求导了。
            pred = model(x)
            preds.append(pred.detach().cpu())  # append方法向list末尾添加元素    detach().cpu()将数据传回CPU用于后续运算
    preds = torch.cat(preds, dim=0).numpy()  # 把Tensor转化为numpy数据类型
    return preds


class COVID19Dataset(Dataset):
    """
    x: Features.
    y: Targets, if none, do prediction.
    Dataset返回一个样本以及标签
    """

    def __init__(self, x, y=None):
        if y is None:
            self.y = y
        else:
            self.y = torch.FloatTensor(y)  # 将list、numpy转换为Tensor类型。
        self.x = torch.FloatTensor(x)

    def __getitem__(self, idx):  # 根据idx返回所对应的样本
        if self.y is None:
            return self.x[idx]
        else:
            return self.x[idx], self.y[idx]

    def __len__(self):  # 返回样本长度
        return len(self.x)


class My_Model(nn.Module):
    def __init__(self, input_dim):
        super(My_Model, self).__init__()
        # TODO: modify model's structure, be aware of dimensions.
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 16),  # （输入维度，输出）
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        x = self.layers(x)
        x = x.squeeze(1)  # (B, 1) -> (B) 删除维度为1的层
        return x


def select_feat(train_data, valid_data, test_data, select_all=True):
    """Selects useful features to perform regression"""
    y_train, y_valid = train_data[:, -1], valid_data[:, -1]
    raw_x_train, raw_x_valid, raw_x_test = train_data[:, :-1], valid_data[:, :-1], test_data

    if select_all:
        feat_idx = list(range(raw_x_train.shape[1]))
    else:
        feat_idx = [0, 1, 2, 3, 4]  # TODO: Select suitable feature columns.

    return raw_x_train[:, feat_idx], raw_x_valid[:, feat_idx], raw_x_test[:, feat_idx], y_train, y_valid


device = 'cuda' if torch.cuda.is_available() else 'cpu'
config = {
    'seed': 5201314,  # Your seed number, you can pick your lucky number. :)
    'select_all': True,  # Whether to use all features.
    'valid_ratio': 0.2,  # validation_size = train_size * valid_ratio
    'n_epochs': 3000,  # Number of epochs.
    'batch_size': 256,
    'learning_rate': 1e-5,
    'early_stop': 400,  # If model has not improved for this many consecutive epochs, stop training.
    'save_path': './models/model.ckpt'  # Your model will be saved here.
}

# DataLoader，训练模型希望传入模型的是minibatches，每个周期完成后希望打乱数据，调用multprocess,即多进程数据
# Set seed for reproducibility
same_seed(config['seed'])

# train_data size: 2699 x 118 (id + 37 states + 16 features x 5 days)
# test_data size: 1078 x 117 (without last day's positive rate)
train_data, test_data = pd.read_csv('./covid.train.csv').values, pd.read_csv('./covid.test.csv').values
train_data, valid_data = train_valid_split(train_data, config['valid_ratio'], config['seed'])

# Print out the data size.
print(f"""train_data size: {train_data.shape} 
valid_data size: {valid_data.shape} 
test_data size: {test_data.shape}""")

# Select features
x_train, x_valid, x_test, y_train, y_valid = select_feat(train_data, valid_data, test_data, config['select_all'])

# Print out the number of features.
print(f'number of features: {x_train.shape[1]}')

train_dataset, valid_dataset, test_dataset = COVID19Dataset(x_train, y_train), \
                                             COVID19Dataset(x_valid, y_valid), \
                                             COVID19Dataset(x_test)

# Pytorch data loader loads pytorch dataset into batches.shuffle：训练完打乱数据集
# 将Dataset塞入Dataloader后进行遍历，可以得到一个batch_size长度的数据
train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
valid_loader = DataLoader(valid_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, pin_memory=True)


def trainer(train_loader, valid_loader, model, config, device):
    criterion = nn.MSELoss(reduction='mean')  # Define your loss function, do not modify this.
    # 'none': no reduction will be applied.
    # 'mean': the sum of the output will be divided by the number of elements in the output.
    # 'sum': the output will be summed.

    # Define your optimization algorithm.
    # TODO: Please check https://pytorch.org/docs/stable/optim.html to get more available algorithms.
    # TODO: L2 regularization (optimizer(weight decay...) or implement by your self).
    optimizer = torch.optim.SGD(model.parameters(), lr=config['learning_rate'], momentum=0.9)
    # 使用SGD会把数据拆分后再分批不断放入 NN 中计算.
    writer = SummaryWriter()  # Writer of tensoboard.将loss可视化

    if not os.path.isdir('./models'):
        os.mkdir('./models')  # Create directory of saving models.

    n_epochs, best_loss, step, early_stop_count = config['n_epochs'], math.inf, 0, 0  # math.inf：无穷大正浮点数

    for epoch in range(n_epochs):
        model.train()  # Set your model to train mode.
        loss_record = []

        # tqdm is a package to visualize your training progress.
        train_pbar = tqdm(train_loader, position=0, leave=True)

        for x, y in train_pbar:
            optimizer.zero_grad()  # Set gradient to zero.
            x, y = x.to(device), y.to(device)  # Move your data to device.
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()  # Compute gradient(backpropagation).
            optimizer.step()  # Update parameters.
            step += 1
            loss_record.append(loss.detach().item()) # 阻断 反向传播并获得标量值

            # Display current epoch number and loss on tqdm progress bar.
            train_pbar.set_description(f'Epoch [{epoch + 1}/{n_epochs}]')
            train_pbar.set_postfix({'loss': loss.detach().item()})

        mean_train_loss = sum(loss_record) / len(loss_record)
        writer.add_scalar('Loss/train', mean_train_loss, step)

        model.eval()  # Set your model to evaluation mode.
        loss_record = []
        for x, y in valid_loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                pred = model(x)
                loss = criterion(pred, y)

            loss_record.append(loss.item())

        mean_valid_loss = sum(loss_record) / len(loss_record)
        print(f'Epoch [{epoch + 1}/{n_epochs}]: Train loss: {mean_train_loss:.4f}, Valid loss: {mean_valid_loss:.4f}')
        writer.add_scalar('Loss/valid', mean_valid_loss, step)

        if mean_valid_loss < best_loss:
            best_loss = mean_valid_loss
            torch.save(model.state_dict(), config['save_path'])  # Save your best model
            print('Saving model with loss {:.3f}...'.format(best_loss))
            early_stop_count = 0
        else:
            early_stop_count += 1

        if early_stop_count >= config['early_stop']:
            print('\nModel is not improving, so we halt the training session.')
            return

# model = My_Model(input_dim=x_train.shape[1]).to(device)  # put your model and data on the same computation device.
# trainer(train_loader, valid_loader, model, config, device)
