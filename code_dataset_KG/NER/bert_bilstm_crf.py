import torch
import torch.nn as nn
from transformers import BertModel

def log_sum_exp_batch(log_Tensor, axis=-1):
    return torch.max(log_Tensor, axis)[0] + \
        torch.log(torch.exp(log_Tensor - torch.max(log_Tensor, axis)[0].view(log_Tensor.shape[0], -1, 1)).sum(axis))

class Bert_BiLSTM_CRF(nn.Module):
    def __init__(self, tag_to_ix, hidden_dim=768):
        super(Bert_BiLSTM_CRF, self).__init__()
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)
        self.hidden_dim = hidden_dim

        self.bert = BertModel.from_pretrained('./bert-base-chinese')
        self.lstm = nn.LSTM(input_size=768, hidden_size=hidden_dim//2, num_layers=2, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim, self.tagset_size)
        self.layernorm = nn.LayerNorm(self.tagset_size)

        # CRF
        self.transitions = nn.Parameter(torch.randn(self.tagset_size, self.tagset_size) * 0.1)
        self.start_label_id = tag_to_ix['[CLS]']
        self.end_label_id = tag_to_ix['[SEP]']
        self.transitions.data[self.start_label_id, :] = -10000
        self.transitions.data[:, self.end_label_id] = -10000

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    def _bert_enc(self, x):
        # 允许微调
        return self.bert(x)[0]

    def _get_lstm_features(self, sentence):
        embeds = self._bert_enc(sentence)
        enc, _ = self.lstm(embeds)
        feats = self.fc(enc)
        feats = self.layernorm(feats)
        return feats

    def _forward_alg(self, feats):
        T = feats.shape[1]
        batch_size = feats.shape[0]
        log_alpha = torch.full((batch_size, 1, self.tagset_size), -10000., device=self.device)
        log_alpha[:, 0, self.start_label_id] = 0.

        for t in range(1, T):
            log_alpha = (log_sum_exp_batch(self.transitions + log_alpha, axis=-1) + feats[:, t]).unsqueeze(1)

        return log_sum_exp_batch(log_alpha)

    def _score_sentence(self, feats, label_ids):
        batch_size, T, _ = feats.shape
        batch_transitions = self.transitions.expand(batch_size, self.tagset_size, self.tagset_size).flatten(1)
        score = torch.zeros((batch_size, 1), device=self.device)
        for t in range(1, T):
            score += batch_transitions.gather(-1, (label_ids[:, t]*self.tagset_size + label_ids[:, t-1]).view(-1,1)) \
                     + feats[:, t].gather(-1, label_ids[:, t].view(-1,1)).view(-1,1)
        return score

    def _viterbi_decode(self, feats):
        T = feats.shape[1]
        batch_size = feats.shape[0]
        log_delta = torch.full((batch_size, 1, self.tagset_size), -10000., device=self.device)
        log_delta[:, 0, self.start_label_id] = 0.
        psi = torch.zeros((batch_size, T, self.tagset_size), dtype=torch.long, device=self.device)

        for t in range(1, T):
            log_delta, psi[:, t] = torch.max(self.transitions + log_delta, -1)
            log_delta = (log_delta + feats[:, t]).unsqueeze(1)

        path = torch.zeros((batch_size, T), dtype=torch.long, device=self.device)
        _, path[:, -1] = torch.max(log_delta.squeeze(), -1)
        for t in range(T-2, -1, -1):
            path[:, t] = psi[:, t+1].gather(-1, path[:, t+1].view(-1,1)).squeeze()
        return path

    def neg_log_likelihood(self, sentence, tags):
        feats = self._get_lstm_features(sentence)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return ((forward_score - gold_score).mean() / feats.shape[1])

    def forward(self, input_ids, attention_mask=None, labels=None):
        """
        input_ids: [batch_size, seq_len]
        attention_mask: [batch_size, seq_len], 1 for real tokens, 0 for padding
        labels: optional, [batch_size, seq_len]
        """
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = bert_output.last_hidden_state  # [batch, seq_len, hidden]

        lstm_out, _ = self.lstm(sequence_output)  # [batch, seq_len, 2*hidden]
        logits = self.fc(lstm_out)  # [batch, seq_len, num_labels]
        logits = self.layernorm(logits)

        if labels is not None:
            # 训练模式，返回 loss
            if attention_mask is not None:
                loss = -self.crf(logits, labels, mask=attention_mask.bool(), reduction='mean')
            else:
                loss = -self.crf(logits, labels, reduction='mean')
            return loss
        else:
            # 预测模式，返回 logits 供 eval 使用
            return logits
