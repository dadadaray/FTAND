# evaluate_cleaning.py
# Python 3.8+
import sys
import re
import json
from collections import Counter, defaultdict
from difflib import SequenceMatcher
import math
import random

def parse_records(text):
    """
    将整段文本分割为记录（假设每条记录以'名称:'或'名称：'开头，字段键格式为 '键名:' 或 '键名：'）
    返回 list of dict
    """
    # 支持中文冒号和英文冒号
    parts = [p.strip() for p in re.split(r'(?=名称[:：])', text) if p.strip()]
    records = []
    # 字段键识别
    keys = ['名称','属什么界','属什么门','属什么纲','属什么目','属什么科','属什么属','属什么种','营养成分']
    for p in parts:
        rec = {}
        for i, k in enumerate(keys):
            # 非贪婪匹配到下一个键或字符串末尾，同时支持中文冒号和英文冒号
            next_keys_pattern = '|'.join(re.escape(x)+r'[:：]' for x in keys if x != k)
            if next_keys_pattern:
                m = re.search(re.escape(k)+r'[:：]\s*(.*?)(?=(?:' + next_keys_pattern + r')|$)', p, re.S)
            else:
                m = re.search(re.escape(k)+r'[:：]\s*(.*)$', p, re.S)
            if m:
                val = m.group(1).strip()
                # 去掉行内换行边缘空白
                val = re.sub(r'\s+', ' ', val).strip()
                rec[k] = val
            else:
                rec[k] = ''
        records.append(rec)
    return records


def safe_tokenize(s):
    # 简单按中文标点/英文标点/空格切分
    tokens = re.split(r'[,\u3000\s;；，、/()（）\-]+', s)
    return [t for t in tokens if t]

def levenshtein_norm(a,b):
    # 使用 SequenceMatcher 的 ratio 作为相似度 -> 转换为距离
    if a==b:
        return 0.0
    return 1.0 - SequenceMatcher(None, a, b).ratio()

def compute_metrics(records_raw, records_clean):
    N_raw = len(records_raw)
    N_clean = len(records_clean)
    keys = ['名称','属什么界','属什么门','属什么纲','属什么目','属什么科','属什么属','属什么种','营养成分']

    # 字段缺失率
    missing_raw = {k:0 for k in keys}
    missing_clean = {k:0 for k in keys}
    for r in records_raw:
        for k in keys:
            if not r.get(k):
                missing_raw[k]+=1
    for r in records_clean:
        for k in keys:
            if not r.get(k):
                missing_clean[k]+=1

    # 重复率（以 名称 为主键）
    names_raw = [r.get('名称','').strip() for r in records_raw]
    names_clean = [r.get('名称','').strip() for r in records_clean]
    dup_rate_raw = (N_raw - len(set(names_raw))) / N_raw if N_raw>0 else 0
    dup_rate_clean = (N_clean - len(set(names_clean))) / N_clean if N_clean>0 else 0

    # 拼写/变更统计：将两文件按名称配对（以 raw 名称为键找最相似的 clean 名称）
    # 构造索引
    clean_name_to_rec = {r.get('名称','').strip(): r for r in records_clean}
    # 对每个 raw 名称，找最相似的 clean 名称（ratio）
    changed_count = 0
    edit_dists = []
    paired = 0
    for r in records_raw:
        raw_name = r.get('名称','').strip()
        if not raw_name:
            continue
        # 找最相似 clean 名称（暴力）
        best = None
        best_ratio = 0.0
        for cn in clean_name_to_rec.keys():
            ratio = SequenceMatcher(None, raw_name, cn).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best = cn
        if best is not None:
            paired += 1
            d = levenshtein_norm(raw_name, best)
            edit_dists.append(d)
            if raw_name != best:
                changed_count += 1

    avg_edit_distance = sum(edit_dists)/len(edit_dists) if edit_dists else 0.0
    proportion_changed = changed_count / paired if paired>0 else 0.0

    # 词汇量与熵（以 营养成分 字段）
    def tokens_stats(records):
        toks = []
        for r in records:
            s = r.get('营养成分','')
            toks += safe_tokenize(s)
        freq = Counter(toks)
        vocab = len(freq)
        total = sum(freq.values()) if freq else 0
        entropy = 0.0
        if total>0:
            for v in freq.values():
                p = v/total
                entropy -= p * math.log2(p)
        return {'vocab':vocab, 'tokens_total':total, 'entropy':entropy, 'top10':freq.most_common(10)}

    toks_raw = tokens_stats(records_raw)
    toks_clean = tokens_stats(records_clean)

    # 非期望字符检测（控制字符或不可见/奇怪符号）
    def noise_rate(records):
        cnt = 0
        for r in records:
            s = ' '.join([r.get(k,'') for k in keys])
            # 控制字符或包含未打印字符
            if re.search(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', s):
                cnt += 1
        return cnt / len(records) if records else 0.0

    noise_raw = noise_rate(records_raw)
    noise_clean = noise_rate(records_clean)

    # 百分比数值解析成功率（示例）
    def parse_percent_rate(records):
        cnt = 0
        for r in records:
            s = r.get('营养成分','')
            if re.search(r'\d+\s*%|\d+％', s):
                cnt += 1
        return cnt / len(records) if records else 0.0

    pct_raw = parse_percent_rate(records_raw)
    pct_clean = parse_percent_rate(records_clean)

    # 汇总
    metrics = {
        'N_raw': N_raw,
        'N_clean': N_clean,
        'missing_raw': missing_raw,
        'missing_clean': missing_clean,
        'dup_rate_raw': dup_rate_raw,
        'dup_rate_clean': dup_rate_clean,
        'avg_edit_distance': avg_edit_distance,
        'proportion_changed': proportion_changed,
        'tokens_raw': toks_raw,
        'tokens_clean': toks_clean,
        'noise_raw': noise_raw,
        'noise_clean': noise_clean,
        'percent_pattern_raw': pct_raw,
        'percent_pattern_clean': pct_clean
    }
    return metrics

def print_report(metrics):
    print("=== Data Cleaning Evaluation Report ===")
    print(f"Records: raw={metrics['N_raw']}  clean={metrics['N_clean']}")
    print("\n-- Missing rates per field (raw -> clean)")
    for k in metrics['missing_raw'].keys():
        mraw = metrics['missing_raw'][k]/metrics['N_raw'] if metrics['N_raw']>0 else 0
        mclean = metrics['missing_clean'][k]/metrics['N_clean'] if metrics['N_clean']>0 else 0
        print(f"{k}: {mraw:.3%} -> {mclean:.3%}")
    print(f"\nDuplicate rate: {metrics['dup_rate_raw']:.3%} -> {metrics['dup_rate_clean']:.3%}")
    print(f"Name change proportion (approx): {metrics['proportion_changed']:.3%}")
    print(f"Average normalized edit distance (name pairs): {metrics['avg_edit_distance']:.4f}")
    print(f"\nNutrition tokens: vocab {metrics['tokens_raw']['vocab']} -> {metrics['tokens_clean']['vocab']}, entropy {metrics['tokens_raw']['entropy']:.4f} -> {metrics['tokens_clean']['entropy']:.4f}")
    print(f"Noise rate (control chars etc): {metrics['noise_raw']:.3%} -> {metrics['noise_clean']:.3%}")
    print(f"Percent-pattern parse rate (e.g. 'xx%'): {metrics['percent_pattern_raw']:.3%} -> {metrics['percent_pattern_clean']:.3%}")
    print("\nTop tokens raw:", metrics['tokens_raw']['top10'])
    print("Top tokens clean:", metrics['tokens_clean']['top10'])

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python evaluate_cleaning.py Raw_dataset.txt Processed_dataset.txt")
        sys.exit(1)
    raw_path = sys.argv[1]
    clean_path = sys.argv[2]
    with open(raw_path, 'r', encoding='utf-8') as f:
        text_raw = f.read()
    with open(clean_path, 'r', encoding='utf-8') as f:
        text_clean = f.read()
    recs_raw = parse_records(text_raw)
    recs_clean = parse_records(text_clean)
    metrics = compute_metrics(recs_raw, recs_clean)
    print_report(metrics)
    # 保存 json 结果
    with open('cleaning_evaluation_result.json','w',encoding='utf-8') as fo:
        json.dump(metrics, fo, ensure_ascii=False, indent=2)
    print("\nSaved numeric results to cleaning_evaluation_result.json")
