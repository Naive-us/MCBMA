# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Config(object):

    """配置参数"""
    def __init__(self, dataset, embedding):
        self.model_name = 'Word2Vec-MCBMA'
        self.train_path = dataset + '/data/train.txt'                                # 训练集
        self.dev_path = dataset + '/data/dev.txt'                                    # 验证集
        self.test_path = dataset + '/data/test.txt'                                  # 测试集
        self.class_list = [x.strip() for x in open(
            dataset + '/data/class.txt', encoding='utf-8').readlines()]              # 类别名单
        self.vocab_path = dataset + '/data/vocab2.pkl'                                # 词表
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'        # 模型训练结果
        self.log_path = dataset + '/log/' + self.model_name
        self.embedding_pretrained = torch.tensor(
            np.load(dataset + '/data/' + embedding)["embeddings"].astype('float32'))\
            if embedding != 'random' else None                                       # 预训练词向量
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备

        self.dropout = 0.5                                              # 随机失活
        self.require_improvement = 1000                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)                         # 类别数
        self.n_vocab = 0                                                # 词表大小，在运行时赋值
        self.num_epochs = 20                                            # epoch数
        self.batch_size = 128                                           # mini-batch大小
        self.pad_size = 80                                              # 每句话处理成的长度(短填长切)
        self.learning_rate = 1e-3                                       # 学习率
        self.embed = self.embedding_pretrained.size(1)\
            if self.embedding_pretrained is not None else 300           # 字向量维度, 若使用了预训练词向量，则维度统一
        self.hidden_size = 256                                          # lstm隐藏层
        self.hidden_size2 = 256
        self.num_layers = 1                                            # lstm层数
        #self.kernel_size = 100

'''Recurrent Convolutional Neural Networks for Text Classification'''


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)
        self.cnn = nn.Conv1d(in_channels=config.embed,
                             out_channels=256,
                             kernel_size=3,
                             stride=1,
                             padding=1)
        self.conv2d = nn.Conv2d(in_channels=config.embed,
                                out_channels=config.hidden_size*3,
                                kernel_size=3,
                                padding=1)

        self.mish1 = Mish()
        self.dropout = nn.Dropout(config.dropout)
 
        self.lstm = nn.LSTM(config.hidden_size2*3, config.hidden_size2*2, config.num_layers,
                            bidirectional=True, batch_first=True, dropout=config.dropout)

        self.mish = nn.GELU()
        self.attention = nn.MultiheadAttention(config.embed, num_heads=2, dropout=0.5)

        self.maxpool = nn.MaxPool1d(kernel_size=config.pad_size)
        self.fc = nn.Linear(config.hidden_size2*2*2+config.hidden_size*3+config.embed, config.num_classes)
        #self.fc2 = nn.Linear(config.hidden_size2, config.num_classes)
        self.w = nn.Parameter(torch.zeros(config.hidden_size2*2))
        self.embed_size =config.hidden_size2*2*2
        self.heads = 8
        self.layers = 1
        self.self_attn = SelfAttention(self.embed_size, self.heads)

        self.input_size =config.hidden_size*3  # 输入特征的维度
        self.hidden_size = config.hidden_size*2  # 隐藏层的维度
        self.num_layers = 1  # 堆叠的层数
        self.batch_first = True  # 是否将batch size作为第一个维度
        self.dropout = nn.Dropout(config.dropout)
        # 实例化Bi-GRU模型
       # self.bigru = BiGRU(self.input_size, self.hidden_size, self.num_layers, self.batch_first)

    def forward(self, x):
        x, _ = x
        embed = self.embedding(x)  # [batch_size, seq_len, embeding]=[64, 32, 64]
        out = embed.unsqueeze(1)#增加维度1
        out = out.permute(0,3,1,2)
       # cnn_out = self.mish(out)
        cnn_out = self.conv2d(out) # [batch_size, hidden_size, seq_len]
        cnn_out = self.mish(cnn_out)
        cnn_out1 = cnn_out.permute(0,3,1,2)
        cnn_out1 = cnn_out1.squeeze()#去掉维度为1

        out, _ = self.lstm(cnn_out1)
        out = self.self_attn(out, out, out, mask=None)

        out = torch.cat((cnn_out1,out,embed), 2)
       # print(out.size())
        out = out.permute(0, 2, 1)
 
       
        out = self.maxpool(out).squeeze()
        out = self.mish(out)
        out = self.fc(out)
        return out

class Mish(nn.Module):
        def forward(self, x):
            return x * torch.tanh(F.softplus(x))

class SwishLayer(nn.Module):
    def __init__(self):
        super(SwishLayer, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Split the embedding into 'heads' number of heads
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        # Einsum does matrix multiplication for query*keys for each training example
        # with a specific head
        attention = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])

        if mask is not None:
            attention = attention.masked_fill(mask == 0, float("-1e20"))

        # Apply softmax activation to the attention scores
        attention = F.softmax(attention / (self.embed_size ** (1 / 2)), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )

        # Combine the attention heads together
        out = self.fc_out(out)

        return out

class TransformerModel(nn.Module):
    def __init__(self, embed_size, heads, layers):
        super(TransformerModel, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.layers = layers
        self.self_attn = SelfAttention(embed_size, heads)
        # 这里可以添加更多的层，例如前馈网络层等
        # ...

    def forward(self, x, mask):
        # 假设x是输入序列，mask是掩码（如果需要的话）
        x = self.self_attn(x, x, x, mask=None)  # 自注意力层的调用
        # ...（可能包括其他层的调用和后续处理）...
        return x


class BiGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch_first=False, dropout=0):
        super(BiGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.dropout = dropout

        # 前向GRU层
        self.gru_fw = nn.GRU(input_size=input_size,
                             hidden_size=hidden_size,
                             num_layers=num_layers,
                             batch_first=batch_first,
                             dropout=(self.dropout if self.num_layers > 1 else 0))

        # 后向GRU层
        self.gru_bw = nn.GRU(input_size=input_size,
                             hidden_size=hidden_size,
                             num_layers=num_layers,
                             batch_first=batch_first,
                             dropout=(self.dropout if self.num_layers > 1 else 0))

    def forward(self, x):
        # 前向传播
        outputs_fw, hidden_fw = self.gru_fw(x)

        # 反向传播，需要将序列倒序
        x_reverse = torch.flip(x, [1])  # 沿着序列维度翻转
        outputs_bw, hidden_bw = self.gru_bw(x_reverse)

        # 将反向传播的结果翻转回原来顺序
        outputs_bw = torch.flip(outputs_bw, [1])

        # 拼接前向和后向的输出
        outputs = torch.cat((outputs_fw, outputs_bw), dim=-1)

        return outputs