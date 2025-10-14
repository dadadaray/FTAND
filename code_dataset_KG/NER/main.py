# -*- coding: utf-8 -*-
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from utils import NerDataset, pad, tokenizer
from bert_bilstm_crf import Bert_BiLSTM_CRF
from bert_cnn_crf import Bert_CNN_CRF
from bert_bilstm_softmax import Bert_BiLSTM_SOFTMAX
from transformers import BertModel
import argparse
from collections import defaultdict

# ----------- 新增: BERT-Softmax & BERT-BiLSTM-Softmax ----------
class BertSoftmax(nn.Module):
    def __init__(self, tag2idx, bert_path="bert-base-chinese", dropout=0.1):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_path)
        hidden = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, len(tag2idx))

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        out = self.dropout(outputs.last_hidden_state)
        return self.classifier(out)


class BertBiLSTMSoftmax(nn.Module):
    def __init__(self, tag2idx, bert_path="bert-base-chinese", hidden_dim=256, dropout=0.1):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_path)
        bert_hidden = self.bert.config.hidden_size
        self.lstm = nn.LSTM(bert_hidden, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, len(tag2idx))

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        out, _ = self.lstm(outputs.last_hidden_state)
        out = self.dropout(out)
        return self.classifier(out)

# ---------------- 训练函数 ----------------
def train(model, iterator, optimizer, criterion, device):
    model.train()
    for i, batch in enumerate(iterator):
        input_ids, label_ids, is_heads, attention_mask = batch
        input_ids, label_ids, is_heads, attention_mask = (
            input_ids.to(device), label_ids.to(device), is_heads.to(device), attention_mask.to(device))
        optimizer.zero_grad()
        if hasattr(model, "neg_log_likelihood"):  # CRF
            loss = model.neg_log_likelihood(input_ids, label_ids)
        else:  # Softmax
            logits = model(input_ids, attention_mask=attention_mask)
            logits_flat = logits.view(-1, logits.size(-1))
            labels_flat = label_ids.view(-1)
            mask_flat = is_heads.view(-1).float()
            loss_all = nn.functional.cross_entropy(logits_flat, labels_flat, reduction='none')
            loss = (loss_all * mask_flat).sum() / mask_flat.sum()
        loss.backward()
        optimizer.step()
        if i % 10 == 0:
            print(f"Step {i}, Loss: {loss.item():.4f}")

# ---------------- 评估函数（Macro / Micro 已分离） ----------------
def eval_model(model, iterator, idx2tag, device, target_tags=None,
               save_dir="results", checkpoint_name="eval_output"):
    import os
    from collections import defaultdict
    import torch

    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    y_true_all, y_pred_all, mask_all = [], [], []
    tag_metrics = defaultdict(lambda: {"true_positive": 0, "false_positive": 0, "false_negative": 0})
    tag_counts = defaultdict(int)
    detailed_lines = []

    with torch.no_grad():
        for batch in iterator:
            input_ids, label_ids, is_heads, attention_mask = batch
            input_ids, label_ids, is_heads, attention_mask = (
                input_ids.to(device), label_ids.to(device),
                is_heads.to(device), attention_mask.to(device)
            )

            if hasattr(model, "crf"):
                _, y_hat = model(input_ids, mask=is_heads.float())
            else:
                logits = model(input_ids, attention_mask=attention_mask)
                y_hat = torch.argmax(logits, dim=-1)

            for b in range(label_ids.size(0)):
                seq_len = int(is_heads[b].sum().item())
                y_true = label_ids[b][:seq_len].cpu().tolist()
                y_pred = y_hat[b][:seq_len].cpu().tolist()
                tokens = input_ids[b][:seq_len].cpu().tolist()

                y_true_all.extend(y_true)
                y_pred_all.extend(y_pred)
                mask_all.extend([1] * seq_len)  # 所有有效token的mask为1

                for tid, t, p in zip(tokens, y_true, y_pred):
                    char = tokenizer.convert_ids_to_tokens([tid])[0]
                    tag_name, pred_name = idx2tag[t], idx2tag[p]
                    detailed_lines.append(f"{char} {tag_name} {pred_name}")

                    # 只统计非特殊标签
                    if tag_name not in ['<PAD>', '[CLS]', '[SEP]']:
                        tag_counts[tag_name] += 1

                        if tag_name == pred_name:
                            tag_metrics[tag_name]["true_positive"] += 1
                        else:
                            tag_metrics[tag_name]["false_negative"] += 1
                            if pred_name not in ['<PAD>', '[CLS]', '[SEP]']:
                                tag_metrics[pred_name]["false_positive"] += 1

    # 过滤特殊标签
    valid_tags = [tag for tag in tag_metrics.keys() if tag not in ['<PAD>', '[CLS]', '[SEP]']]

    # 1. 逐类别计算 P、R、F1（Macro计算用）
    tag_prf = {}
    for tag in valid_tags:
        m = tag_metrics[tag]
        tp, fp, fn = m["true_positive"], m["false_positive"], m["false_negative"]
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
        tag_prf[tag] = (p, r, f1)

    # 2. Macro 总体 = 各类别指标的算术平均
    macro_precision = sum(v[0] for v in tag_prf.values()) / len(tag_prf) if tag_prf else 0.0
    macro_recall = sum(v[1] for v in tag_prf.values()) / len(tag_prf) if tag_prf else 0.0
    macro_f1 = sum(v[2] for v in tag_prf.values()) / len(tag_prf) if tag_prf else 0.0

    # 3. Micro 总体 = 直接计算总体TP、FP、FN
    total_tp = sum(tag_metrics[tag]["true_positive"] for tag in valid_tags)
    total_fp = sum(tag_metrics[tag]["false_positive"] for tag in valid_tags)
    total_fn = sum(tag_metrics[tag]["false_negative"] for tag in valid_tags)

    micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = (2 * micro_p * micro_r / (micro_p + micro_r)) if (micro_p + micro_r) > 0 else 0.0

    # ----------- 保存结果 -----------
    detailed_file = os.path.join(save_dir, f"{checkpoint_name}_detailed.txt")
    with open(detailed_file, "w", encoding="utf-8") as f:
        f.write("\n".join(detailed_lines))
    print(f"逐字预测结果已保存到: {detailed_file}")

    final_file = os.path.join(save_dir, f"{checkpoint_name}_metrics.txt")
    with open(final_file, "w", encoding="utf-8") as fout:
        fout.write(f"Macro 总体 精确度={macro_precision:.6f}\n")
        fout.write(f"Macro 总体 召回率={macro_recall:.6f}\n")
        fout.write(f"Macro 总体 F1={macro_f1:.6f}\n")
        fout.write(f"Micro 总体 精确度={micro_p:.6f}\n")
        fout.write(f"Micro 总体 召回率={micro_r:.6f}\n")
        fout.write(f"Micro 总体 F1={micro_f1:.6f}\n\n")

        if target_tags is None:
            target_tags = [tag for tag in valid_tags if tag not in ['<PAD>', '[CLS]', '[SEP]']]

        fout.write("指定标签的精确度、召回率、F1和数量:\n")
        for tag in target_tags:
            if tag in tag_prf:
                precision, recall, f1 = tag_prf[tag]
                count = tag_counts.get(tag, 0)
                fout.write(f"{tag} - 精确度: {precision:.6f}, 召回率: {recall:.6f}, F1: {f1:.6f}, 数量: {count}\n")
                print(f"{tag} - 精确度: {precision:.6f}, 召回率: {recall:.6f}, F1: {f1:.6f}, 数量: {count}")

    print(f"\nMacro 总体 - 精确度: {macro_precision:.6f}, 召回率: {macro_recall:.6f}, F1: {macro_f1:.6f}")
    print(f"Micro 总体 - 精确度: {micro_p:.6f}, 召回率: {micro_r:.6f}, F1: {micro_f1:.6f}")

    return macro_precision, macro_recall, macro_f1, micro_p, micro_r, micro_f1
# ---------------- 主程序 ----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--model_type", type=str,
                        choices=["bilstm_crf", "cnn_crf", "bilstm_softmax", "bert_softmax", "bert_bilstm_softmax"],
                        required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--trainset", type=str, default=None)
    parser.add_argument("--validset", type=str, required=True)
    parser.add_argument("--logdir", type=str, default="checkpoints")
    parser.add_argument("--n_epochs", type=int, default=30)
    parser.add_argument("--only_eval", action="store_true")
    hp = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # tag2idx
    if hp.checkpoint:
        tag_path = hp.checkpoint.replace(".pt", "_tag2idx.json")
        if os.path.exists(tag_path):
            with open(tag_path, "r", encoding="utf-8") as f:
                tag2idx = json.load(f)
        else:
            raise FileNotFoundError(f"{tag_path} 不存在，请先保存训练时的 tag2idx.json")
    else:
        VOCAB = ('<PAD>', '[CLS]', '[SEP]', 'O', 'B-NUT', 'I-NUT', 'B-FOOD', 'I-FOOD')
        tag2idx = {t: i for i, t in enumerate(VOCAB)}
    idx2tag = {v: k for k, v in tag2idx.items()}

    # 模型初始化
    if hp.model_type == "bilstm_crf":
        model = Bert_BiLSTM_CRF(tag2idx).to(device)
    elif hp.model_type == "cnn_crf":
        model = Bert_CNN_CRF(tag2idx).to(device)
    elif hp.model_type == "bilstm_softmax":
        model = Bert_BiLSTM_SOFTMAX(tag2idx).to(device)
    elif hp.model_type == "bert_softmax":
        model = BertSoftmax(tag2idx).to(device)
    else:  # bert_bilstm_softmax
        model = BertBiLSTMSoftmax(tag2idx).to(device)

    # 加载 checkpoint
    if hp.checkpoint and os.path.exists(hp.checkpoint):
        print(f"从 {hp.checkpoint} 加载模型参数...")
        model.load_state_dict(torch.load(hp.checkpoint, map_location=device))
        print("模型初始化完成。")

    # 数据
    if hp.trainset:
        train_dataset = NerDataset(hp.trainset, tokenizer, tag2idx)
        train_iter = data.DataLoader(train_dataset, batch_size=hp.batch_size, shuffle=True, collate_fn=pad)
    eval_dataset = NerDataset(hp.validset, tokenizer, tag2idx)
    eval_iter = data.DataLoader(eval_dataset, batch_size=hp.batch_size, shuffle=False, collate_fn=pad)

    # ---------------- 训练 ----------------
    if not hp.only_eval and hp.trainset:
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        criterion = nn.CrossEntropyLoss(ignore_index=tag2idx["<PAD>"])
        for epoch in range(1, hp.n_epochs + 1):
            print(f"\n===== Epoch {epoch} =====")
            train(model, train_iter, optimizer, criterion, device)
        os.makedirs(hp.logdir, exist_ok=True)
        final_ckpt = os.path.join(hp.logdir, f"{hp.model_type}_final.pt")
        torch.save(model.state_dict(), final_ckpt)
        with open(final_ckpt.replace(".pt", "_tag2idx.json"), "w", encoding="utf-8") as f:
            json.dump(tag2idx, f, ensure_ascii=False, indent=2)
        print(f"训练完成，最终模型已保存到 {final_ckpt}")

    # ---------------- 评估 ----------------
    print(f"\n开始评估 {len(eval_dataset)} 条样本...")
    eval_model(model, eval_iter, idx2tag, device, save_dir=hp.logdir)