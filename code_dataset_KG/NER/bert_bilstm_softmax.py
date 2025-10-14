# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from transformers import BertModel

class Bert_BiLSTM_SOFTMAX(nn.Module):
    def __init__(self, tag2idx, bert_model_path='./bert-base-chinese', lstm_hidden=256, lstm_layers=1, dropout=0.1):
        super(Bert_BiLSTM_SOFTMAX, self).__init__()
        self.tag2idx = tag2idx
        self.tagset_size = len(tag2idx)

        # BERT 编码器
        self.bert = BertModel.from_pretrained(bert_model_path)
        self.bert_hidden = self.bert.config.hidden_size

        # BiLSTM 层
        self.lstm = nn.LSTM(
            input_size=self.bert_hidden,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0
        )

        # 线性层映射到 tag 数量
        self.hidden2tag = nn.Linear(lstm_hidden * 2, self.tagset_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask=None):
        # BERT 输出
        outputs = self.bert(input_ids, attention_mask=attention_mask)[0]  # [B, L, H]

        # BiLSTM
        lstm_out, _ = self.lstm(outputs)  # [B, L, 2*hidden]
        lstm_out = self.dropout(lstm_out)

        # 线性映射到标签空间
        logits = self.hidden2tag(lstm_out)  # [B, L, tagset_size]

        if self.training:
            # 训练模式下返回 logits，用于 loss
            return logits
        else:
            # 评估模式下直接返回预测标签 Tensor
            y_hat = torch.argmax(logits, dim=-1)  # [B, L]
            return y_hat
