# -*- coding: utf-8 -*-
import os
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer

# ---------------- 配置 ----------------
VOCAB = ('<PAD>', '[CLS]', '[SEP]', 'O', 'B-NUT', 'I-NUT', 'B-FOOD', 'I-FOOD')
tag2idx = {tag: idx for idx, tag in enumerate(VOCAB)}
idx2tag = {idx: tag for idx, tag in enumerate(VOCAB)}

# Bert tokenizer
BERT_PATH = "/home/yanrongen/YingyangNER/bert-base-chinese"   # 指向你放文件的目录
tokenizer = BertTokenizer.from_pretrained(BERT_PATH)

# ---------------- 数据集 ----------------
class NerDataset(Dataset):
    def __init__(self, file_path, tokenizer, tag2idx, max_len=512, stride=128):
        """
        file_path: 数据文件路径
        tokenizer: BERT分词器
        tag2idx: 标签到索引映射
        max_len: BERT输入最大长度 (含CLS和SEP)，默认512
        stride: 滑动窗口重叠步长
        """
        self.samples = []
        self.tokenizer = tokenizer
        self.tag2idx = tag2idx
        self.max_len = max_len
        self.stride = stride

        with open(file_path, 'r', encoding='utf-8') as f:
            words, labels = [], []
            for line in f:
                line = line.strip()
                if not line:
                    if words:
                        self._process_and_split(words, labels)
                        words, labels = [], []
                    continue
                splits = line.split()
                words.append(splits[0])
                labels.append(splits[-1])
            if words:
                self._process_and_split(words, labels)

    def _process_and_split(self, words, labels):
        """把一条样本拆成多个 chunk，每个 chunk <= max_len"""
        tokens_all, labels_all, is_heads_all = [], [], []

        for w, l in zip(words, labels):
            sub_tokens = self.tokenizer.tokenize(w) if w not in ('[CLS]', '[SEP]') else [w]
            tokens_all.extend(sub_tokens)
            labels_all.extend([self.tag2idx.get(l, self.tag2idx['O'])] +
                              [self.tag2idx['<PAD>']] * (len(sub_tokens)-1))
            is_heads_all.extend([1] + [0]*(len(sub_tokens)-1))

        max_len = self.max_len - 2  # 留给 CLS 和 SEP
        start = 0
        while start < len(tokens_all):
            end = min(start + max_len, len(tokens_all))
            chunk_tokens = ['[CLS]'] + tokens_all[start:end] + ['[SEP]']
            chunk_labels = [self.tag2idx['[CLS]']] + labels_all[start:end] + [self.tag2idx['[SEP]']]
            chunk_heads = [1] + is_heads_all[start:end] + [0]

            # 补齐
            pad_len = self.max_len - len(chunk_tokens)
            chunk_tokens += ['[PAD]'] * pad_len
            chunk_labels += [self.tag2idx['<PAD>']] * pad_len
            chunk_heads += [0] * pad_len

            input_ids = self.tokenizer.convert_tokens_to_ids(chunk_tokens)
            attention_mask = [1 if id != self.tokenizer.pad_token_id else 0 for id in input_ids]

            self.samples.append((torch.tensor(input_ids),
                                 torch.tensor(chunk_labels),
                                 torch.tensor(chunk_heads),
                                 torch.tensor(attention_mask)))
            if end == len(tokens_all):
                break
            start += max_len - self.stride

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

# ---------------- pad 函数 ----------------
def pad(batch):
    input_ids = torch.stack([b[0] for b in batch])
    label_ids = torch.stack([b[1] for b in batch])
    is_heads = torch.stack([b[2] for b in batch])
    attention_mask = torch.stack([b[3] for b in batch])
    return input_ids, label_ids, is_heads, attention_mask
