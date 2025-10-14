import torch
import torch.nn as nn
from transformers import BertModel

class BertSoftmax(nn.Module):
    def __init__(self, bert_path, num_labels, dropout=0.1):
        super(BertSoftmax, self).__init__()
        self.bert = BertModel.from_pretrained(bert_path)
        hidden_size = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = self.dropout(outputs.last_hidden_state)  # (batch, seq, hidden)
        logits = self.classifier(sequence_output)  # (batch, seq, num_labels)
        return logits
