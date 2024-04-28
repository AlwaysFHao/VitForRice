import math

import numpy as np
import torch
import torch.nn as nn
import warnings

from torch.nn import init


def get_pad_attention_mask(Q_seq, K_seq, pad_index_in_vocab):
    """
    获取针对pad符号的attention_mask
    :param Q_seq: [batch_size, q_seq_len]
    :param K_seq: [batch_size, k_seq_len]
    :param pad_index_in_vocab: pad在字典中的位置
    :return: attention_mask [batch_size, q_seq_len, k_seq_len],对于K_seq中的pad添加mask
    """
    # 获取batch_size和序列长度
    batch_size, q_seq_len = Q_seq.shape[0], Q_seq.shape[1]
    k_seq_len = K_seq.shape[1]

    # 根据pad在字典中索引构建注意力掩码
    # [batch_size, k_seq_len] -> [batch_size, 1, k_seq_len]
    pad_attention_mask = K_seq.detach().eq(pad_index_in_vocab).unsqueeze(1)
    # [batch_size, 1, k_seq_len] -> [batch_size, q_seq_len, k_seq_len]
    return pad_attention_mask.expand(batch_size, q_seq_len, k_seq_len)


def get_all_zero_attention_mask(batch_size, q_seq_len, k_seq_len):
    """
    构造全零注意力掩码
    :param batch_size: 批处理大小
    :param q_seq_len: q序列长度
    :param k_seq_len: k序列长度
    :return: attention_mask [batch_size, q_seq_len, k_seq_len]
    """
    # 构造全零注意力掩码
    all_zero_attention_mask = torch.zeros((batch_size, q_seq_len, k_seq_len), dtype=torch.bool)
    return all_zero_attention_mask


def get_subsequent_attention_mask(seq):
    """
    构造decoder时的屏蔽下文attention_mask
    :param seq: [batch_size, seq_len]
    :return: subsequence_mask [batch_size, seq_len, seq_len]
    """
    # [batch_size, seq_len, seq_len]
    attention_mask_shape = [seq.shape[0], seq.shape[1], seq.shape[1]]
    # 生成一个严格上三角矩阵
    # np.triu()返回一个上三角矩阵，自对角线k以下元素全部置为0，k代表对角线上下偏移程度，这里将k设置为1是为了构建严格上三角矩阵（对角线全为0）
    subsequence_mask = np.triu(np.ones(attention_mask_shape), k=1)
    # 如果没转成byte，这里默认是Double(float64)，占据的内存空间大，浪费，用byte就够了
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()
    return subsequence_mask


class PositionalEncoding(nn.Module):
    """
    位置编码层
    """

    def __init__(self,
                 embedding_dim,
                 seq_len):
        """
        构造方法
        :param embedding_dim: 词向量维度
        :param seq_len: 序列长度
        """
        super(PositionalEncoding, self).__init__()
        # 以下为transformer的positionalEncoding公式
        # 第pos个词向量偶数位置： PE_{(pos, 2i)} = \sin (\frac{pos}{10000^{\frac{2i}{d_{k}}}})
        # 第pos个词向量奇数位置： PE_{(pos, 2i + 1)} = \cos (\frac{pos}{10000^{\frac{2i}{d_{k}}}})
        # 其中 \frac{1}{10000^{\frac{2i}{d_{k}}}} 可以等价替换为 e^{-\frac{2i}{d} ln(10000)}
        positional_encoding = torch.zeros(seq_len, embedding_dim)
        # 生成位置（这里添加一个维度在前面是为了后面批次计算）[seq_len] -> [seq_len, 1]
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        # e^{-\frac{2i}{d_k} ln(10000)} shape: [embedding_dim / 2]
        div_term = torch.exp(-(torch.arange(0, embedding_dim, 2).float() / embedding_dim) * (math.log(10000.0)))
        # 偶数部分
        # position[seq_len, 1] * div_term[embedding_dim / 2]
        # -> 广播机制变为 [seq_len, embedding_dim / 2] /dot [seq_len, embedding_dim / 2]
        positional_encoding[:, 0::2] = torch.sin(position * div_term)
        # 奇数部分
        if embedding_dim > 1:
            if embedding_dim % 2 == 0:
                positional_encoding[:, 1::2] = torch.cos(position * div_term)
            else:
                positional_encoding[:, 1::2] = torch.cos(position * div_term[0:-1])

        # 定义为固定参数
        self.register_buffer('pe', positional_encoding)

    def forward(self, seq):
        # 获取当前序列长度，并叠加上位置编码
        # 假定这里x输入的shape为 [batch_size, seq_len, embedding_dim]
        # 广播机制self.positional_encoding shape变为[batch_size, seq_len, embedding_dim]
        return seq + self.pe[:seq.shape[1], :]


class PositionalParameter(nn.Module):
    """
    可学习参数位置编码
    """

    def __init__(self, embedding_dim, seq_len):
        """
        构造方法
        :param embedding_dim: 词嵌入维度
        :param seq_len: 序列长度
        """
        super(PositionalParameter, self).__init__()
        self.pos_embedding = nn.Parameter(torch.empty(1, seq_len, embedding_dim).normal_(std=0.02))

    def forward(self, x):
        """
        前向传播
        :param x: 输入数据 shape: [batch, seq_len, embedding_dim]
        :return:
        """
        return x + self.pos_embedding


class ScaledDotProductAttention(nn.Module):
    """
    点积缩放
    softmax(\frac{Q @ K^T}{ sqrt{d_k}}) @ V
    """

    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, Q, K, V, attention_mask):
        """
        前向传播
        softmax(\frac{Q @ K^T}{ sqrt{d_k}}) @ V
        :param Q: Q: Query [batch_size, num_heads, q_seq_len, q_dim]
        :param K: K: Key [batch_size, num_heads, k_seq_len, k_dim]
        :param V: V: Value [batch_size, num_heads, v_seq_len, v_dim]
        :param attention_mask: [batch_size, num_heads, q_seq_len, k_seq_len]
        :return: (context: [batch_size, num_heads, seq_len, v_dim],
                    attention: [batch_size, num_heads, q_seq_len, k_seq_len])
        """
        # 输入数据校验
        # 由于要进行Q @　K^T， 因此q_dim必须等于k_dim
        assert Q.shape[-1] == K.shape[-1], 'Q, K输入词向量维度必须相同！'
        # Q @ K^T 得到的结果shape为[batch_size, num_heads, q_seq_len, k_seq_len]，后续再与V做矩阵乘法，
        # 因此k_seq_len必须等于v_seq_dim
        assert K.shape[-1] == V.shape[-1], 'K, V输入的序列长度必须相同！'
        # 由于attention_mask作用在 Q @ K^T 上，因此要求attention_mask尺寸必须是[batch_size, num_heads, q_seq_len, k_seq_len]
        assert Q.shape[-2] == attention_mask.shape[-2] and K.shape[-2] == attention_mask.shape[-1], \
            f'attention_mask尺寸{attention_mask.shape} != ' \
            f'Q @　K^T 的尺寸{[Q.shape[0], Q.shape[1], Q.shape[2], K.shape[-2]]}!'

        # 获取词向量维度做缩放
        d_k = Q.shape[-1]
        # Q @ K^T / sqrt(d_k)
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)
        # 添加mask，将mask位置设置为无限小，通过softmax基本就是0，最后打掩码位置就不会对结果产生影响
        scores.masked_fill_(attention_mask.to(scores.device), -1e20)
        # softmax(\frac{Q @ K^T}{ sqrt{d_k}})
        attention = self.softmax(scores)
        # softmax(\frac{Q @ K^T}{ sqrt{d_k}}) @ V
        context = torch.matmul(attention, V)
        # 返回结果和注意力矩阵
        return context, attention


class MultiHeadSelfAttention(nn.Module):
    """
    多头注意力层
    """

    def __init__(self,
                 embedding_dim,
                 k_dim,
                 v_dim,
                 num_heads, ):
        """
        构造方法
        :param embedding_dim: 词嵌入维度
        :param k_dim: 最后输出的K的维度
        :param v_dim: 最后输出的V的维度
        :param num_heads: 总共要几个头
        """
        super(MultiHeadSelfAttention, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.k_dim = k_dim
        self.v_dim = v_dim
        # 定义生成QKV矩阵的线性层
        # 注意这里考虑多头注意力，因此实际输出向量长度为原维度的num_heads倍，后面再拆分
        self.W_Q = nn.Linear(embedding_dim, k_dim * num_heads)
        self.W_K = nn.Linear(embedding_dim, k_dim * num_heads)
        self.W_V = nn.Linear(embedding_dim, v_dim * num_heads)
        # 多头结果拼接融合层
        self.fc = nn.Linear(v_dim * num_heads, embedding_dim)
        # ScaledDotProductAttention
        self.scaled_dot_product_attention = ScaledDotProductAttention()

    def forward(self, Q, K, V, attention_mask):
        """
        前向传播计算
        :param Q: Query [batch_size, q_seq_len, q_embedding_dim]
        :param K: Key [batch_size, k_seq_len, k_embedding_dim]
        :param V: Value [batch_size, v_seq_len, v_embedding_dim]
        :param attention_mask: 掩码，用来标记padding等 [batch_size, q_seq_len, k_seq_len]
        :return: (self-attention结果: [batch_size, seq_len, embedding_dim],
                    attention: [batch_size, num_heads, q_seq_len, k_seq_len])
        """
        assert Q.shape[-1] == K.shape[-1], 'Q, K输入词向量维度必须相同！'

        # 获取残差连接输入和batch_size
        residual, batch_size = V, V.shape[0]

        # 得到QKV，并拆分为多头[batch_size, seq_len, num_heads, k or v dim]
        # 然后transpose成[batch_size, num_heads, seq_len, k or v dim]
        q_s = self.W_Q(Q).view(batch_size, -1, self.num_heads, self.k_dim).transpose(1, 2)
        k_s = self.W_K(K).view(batch_size, -1, self.num_heads, self.k_dim).transpose(1, 2)
        v_s = self.W_V(V).view(batch_size, -1, self.num_heads, self.v_dim).transpose(1, 2)

        # attention_mask
        # [batch_size, q_seq_len, k_seq_len] -> [batch_size, 1, q_seq_len, k_seq_len]
        # -> [batch_size, num_heads, q_seq_len, k_seq_len]
        attention_mask = attention_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)

        # 通过ScaledDotProductAttention聚合上下文信息
        context, attention = self.scaled_dot_product_attention(q_s, k_s, v_s, attention_mask)
        # 首先通过transpose方法转置 [batch_size, num_heads, seq_len, v_dim] -> [batch_size, seq_len, num_heads, v_dim]
        # 然后通过contiguous解决转置带来的非连续存储问题，提升性能
        # 之后再用将多头信息concat到一起[batch_size, seq_len, num_heads, v_dim] -> [batch_size, seq_len, num_heads * v_dim]
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.v_dim)
        # 线性变换融合多头信息 [batch_size, seq_len, num_heads * v_dim] -> [batch_size, seq_len, embedding_dim]
        output = self.fc(context)
        # Add
        output += residual
        return output, attention


class PositionWiseFeedForward(nn.Module):
    """
    前馈神经网络
    Position-wise意为对每个点独立做，即对序列中的每个token独立过同一个MLP，即作用在输入的最后一个维度上
    这里可选Conv1D和Linear两种实现方式，Linear考虑全局，而Conv1D则是只考虑邻近特征
    """

    def __init__(self, input_size,
                 hidden_size,
                 mode: str):
        """
        构造方法
        :param input_size: 词向量维度
        :param hidden_size: 隐藏层维度
        :param mode: 使用 linear or conv
        """
        super(PositionWiseFeedForward, self).__init__()
        assert mode in ['linear', 'conv'], "mode 必须是 'linear' or 'conv'!\a"
        self.__mode = mode
        if self.__mode == 'linear':
            self.fc = nn.Sequential(
                nn.Linear(in_features=input_size, out_features=hidden_size, bias=False),
                nn.GELU(),
                nn.Linear(in_features=hidden_size, out_features=input_size, bias=False)
            )
        elif self.__mode == 'conv':
            self.conv = nn.Sequential(
                nn.Conv1d(in_channels=input_size, out_channels=hidden_size, kernel_size=3, padding=1, bias=False),
                nn.GELU(),
                nn.Conv1d(in_channels=hidden_size, out_channels=input_size, kernel_size=3, padding=1, bias=False)
            )

    def forward(self, x):
        """
        前向传播
        :param x: [batch_size, seq_len, embedding_dim]

        :return: x + W @ x
        """
        residual = x
        if self.__mode == 'linear':
            return residual + self.fc(x)
        elif self.__mode == 'conv':
            # x 需要转置一下，让卷积在seq_len维度进行卷积
            return residual + self.conv(x.transpose(1, 2)).transpose(1, 2)
        else:
            raise ValueError("mode 必须是 'linear' or 'conv'!\a")


class EncodeLayer(nn.Module):
    """
    Encoder块
    包含自注意力层和前馈神经网络层
    """

    def __init__(self,
                 embedding_dim,
                 k_dim,
                 v_dim,
                 num_heads,
                 ffn_hidden_size,
                 ffn_mode,
                 drop_out_radio,
                 is_norm_first: bool = True):
        """
        构造方法
        :param embedding_dim: 词嵌入维度
        :param k_dim: 最后输出的K的维度
        :param v_dim: 最后输出的V的维度
        :param ffn_hidden_size: 前馈神经网络的隐藏层神经元结点个数
        :param ffn_mode: 前馈神经网络的模式，可选 'linear' 和 'conv'
        :param num_heads: 总共要几个头
        :param drop_out_radio: dropout概率
        :param is_norm_first: 是否先做norm
        """
        super(EncodeLayer, self).__init__()
        # 实例化多头注意力层
        self.encode_self_attention = MultiHeadSelfAttention(embedding_dim, k_dim, v_dim, num_heads)
        self.ln_1 = nn.LayerNorm(embedding_dim)
        self.dropout1 = nn.Dropout(drop_out_radio)
        self.feed_forward = PositionWiseFeedForward(embedding_dim, ffn_hidden_size, mode=ffn_mode)
        self.ln_2 = nn.LayerNorm(embedding_dim)
        self.dropout2 = nn.Dropout(drop_out_radio)
        self.is_norm_first = is_norm_first

    def forward(self, inputs, attention_mask):
        if self.is_norm_first:
            # 先做norm
            inputs = self.ln_1(inputs)
            output, attention = self.encode_self_attention(Q=inputs, K=inputs, V=inputs, attention_mask=attention_mask)
            output = self.dropout1(output)

            # 先做norm
            output = self.ln_2(output)
            output = self.feed_forward(output)
            output = self.dropout2(output)
        else:
            output, attention = self.encode_self_attention(Q=inputs, K=inputs, V=inputs, attention_mask=attention_mask)
            # 后做norm
            output = self.ln_1(output)
            output = self.dropout1(output)

            output = self.feed_forward(output)
            # 后做norm
            output = self.ln_2(output)
            output = self.dropout2(output)

        return output, attention


class Encoder(nn.Module):
    """
    Encoder
    """

    def __init__(self,
                 embedding_dim: int,
                 num_heads: int,
                 seq_len: int,
                 num_layers: int,
                 pos_embedding_is_Parameter: bool = True,
                 k_dim: int = 0,
                 v_dim: int = 0,
                 ffn_hidden_size: int = 0,
                 ffn_mode: str = 'linear',
                 is_norm_first: bool = True,
                 drop_out_radio: float = 0.1):
        """
        构造方法
        :param embedding_dim: 词嵌入维度
        :param num_heads: 多头注意力头数
        :param seq_len: 序列长度
        :param num_layers: encoder块的层数
        :param pos_embedding_is_Parameter: 位置编码是否为可学习参数
        :param k_dim: 单头自注意力key的维度
        :param v_dim: 单头自注意力value的维度
        :param ffn_hidden_size: 前馈神经网络隐藏层大小
        :param ffn_mode: 前馈神经网络模式，默认为线性
        :param is_norm_first: 是否前置norm层
        :param drop_out_radio: dropout概率
        """
        super(Encoder, self).__init__()

        # 如果没有给定前馈神经网络的隐藏层参数就直接等于embedding_dim
        if ffn_hidden_size == 0:
            ffn_hidden_size = embedding_dim
        if k_dim == 0:
            k_dim = embedding_dim
        if v_dim == 0:
            v_dim = embedding_dim

        # 位置编码层
        if pos_embedding_is_Parameter:
            self.pos_embedding = PositionalParameter(embedding_dim, seq_len)
        else:
            self.pos_embedding = PositionalEncoding(embedding_dim, seq_len)
        self.ln = nn.LayerNorm(embedding_dim)
        # 创建encoder块
        self.encoder_layers = nn.ModuleList([EncodeLayer(embedding_dim=embedding_dim,
                                                         k_dim=k_dim,
                                                         v_dim=v_dim,
                                                         num_heads=num_heads,
                                                         ffn_hidden_size=ffn_hidden_size,
                                                         ffn_mode=ffn_mode,
                                                         drop_out_radio=drop_out_radio,
                                                         is_norm_first=is_norm_first) for _ in range(num_layers)])

    def forward(self, x):
        """
        前向传播
        :param x: [batch_size, seq_len] or [batch_size, seq_len, embedding_dim]
        :return: (output: [batch_size, seq_len, embedding_dim])
        """

        # 设置全0mask
        attention_mask = get_all_zero_attention_mask(x.shape[0], x.shape[1], x.shape[1])
        # 叠加位置编码
        out_put = self.pos_embedding(x)

        # 创建各层注意力权重列表
        encoder_self_attentions = []

        # 开始输入encoder块
        for encoder_layer in self.encoder_layers:
            # 依次丢进encoder块中
            out_put, attention = encoder_layer(out_put, attention_mask)
            # 将注意力权重加入列表保存
            encoder_self_attentions.append(attention)

        return self.ln(out_put), encoder_self_attentions


class VisionTransformer(nn.Module):
    def __init__(self,
                 image_size: int,
                 patch_size: int,
                 embedding_dim: int,
                 num_classes: int,
                 num_heads: int,
                 num_layers: int,
                 pos_embedding_is_Parameter: bool = True,
                 representation_size: int = 0,
                 k_dim: int = 0,
                 v_dim: int = 0,
                 ffn_hidden_size: int = 0,
                 ffn_mode: str = 'linear',
                 is_norm_first: bool = True,
                 drop_out_radio: float = 0.1):
        """
        Vit
        :param image_size: 图像尺寸
        :param patch_size: patch块大小
        :param embedding_dim: 词向量维度
        :param num_classes: 最后分类类别
        :param num_heads: 多头注意力头数
        :param num_layers: encoder块层数
        :param pos_embedding_is_Parameter: 位置编码是否使用可学习参数
        :param representation_size: 多分类层表征维度
        :param k_dim: 单头key维度
        :param v_dim: 单头value维度
        :param ffn_hidden_size: ffn维度
        :param ffn_mode: ffn模式
        :param is_norm_first: 是否前置norm层
        :param drop_out_radio: drop_out率
        """
        super(VisionTransformer, self).__init__()
        # 判断是否可以整除
        if image_size % patch_size != 0:
            raise ValueError('Image size must be divisible by patch size!')

        if representation_size == 0:
            representation_size = embedding_dim
        # Patch-Linear
        # need_shape: [batch, seq_len, embedding_dim]
        # [batch, 3, 224, 224] -> [batch, 768, 14, 14]
        # seq_len = (h / patch_size)^2
        # [batch, 768, 196] -> [batch, 196, 768]
        self.conv_proj = nn.Conv2d(
            in_channels=3, out_channels=embedding_dim, kernel_size=patch_size, stride=patch_size
        )
        self.bn = nn.BatchNorm2d(embedding_dim)

        # 计算序列长度
        seq_len = (image_size // patch_size) ** 2

        # 添加class token
        # [batch, 196, 768] -> [batch, 197, 768]
        # [batch, 1, 768]
        self.class_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))
        seq_len += 1

        # encoder层
        self.encoder = Encoder(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            seq_len=seq_len,
            num_layers=num_layers,
            pos_embedding_is_Parameter=pos_embedding_is_Parameter,
            k_dim=k_dim,
            v_dim=v_dim,
            ffn_hidden_size=ffn_hidden_size,
            ffn_mode=ffn_mode,
            is_norm_first=is_norm_first,
            drop_out_radio=drop_out_radio)

        # 最后分类头
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, representation_size),
            nn.Tanh(),
            nn.Linear(representation_size, num_classes)
        )
        # 初始化
        self.init_params()

    def init_params(self):
        """
        初始化参数，线形层用xavier初始化，卷积用何凯明初始化
        :return:
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)

    def forward(self, x):
        batch = x.shape[0]
        x = torch.flatten(self.bn(self.conv_proj(x)), start_dim=-2).transpose(-1, -2)

        # [1, 1, embedding_dim] -> [batch, 1, embedding_dim]
        batch_class_token = self.class_token.expand(batch, -1, -1)
        # [batch, 196, 768] -> [batch, 197, 768]
        x = torch.cat([batch_class_token, x], dim=1)

        x, _ = self.encoder(x)

        # 取出cls_token
        x = x[:, 0, :]
        # [batch, embedding_dim] -> [batch, num_classes]
        x = self.classifier(x)

        return x


if __name__ == '__main__':
    data = torch.randn(8, 3, 224, 224)
    model = VisionTransformer(
        image_size=224,
        patch_size=16,
        embedding_dim=768,
        num_classes=10,
        num_heads=8,
        num_layers=6)
    print(model(data).shape)
