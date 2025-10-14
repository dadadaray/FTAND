# FoodKG: A Dataset and Knowledge Graph for Forest Foods & Nutrition

> **Purpose:** This README provides English-language documentation, clear file and folder descriptions, and a step-by-step usage guide for users.

---

## Overview

FoodKG is a dataset and codebase for constructing a knowledge graph about forest foods and their nutritional components.

It integrates data crawling, named entity recognition (NER), relationship extraction, and structured dataset generation in both English and Chinese.

The project is implemented primarily in Python, using BERT-based NER models and Neo4j for knowledge graph construction.

This document includes:

* A clear description of repository layout and file contents
* Dependencies and environment setup
* Step-by-step reproduction instructions (data processing → NER → Bert experiments → Neo4j import)
* Example commands and Cypher queries

---

## Repository structure

```
code_dataset_KG/
├─ dataset/
│  ├─ Chinese relation data/     
│  │      Prosessed_dataset.txt
│  │      Raw_dataset.txt
│  │      属.csv / 属于关系.xlsx / 目.csv / 种.csv / 科.csv / 纲.csv / 门.csv/关系抽取后总关系.csv/含有关系.xlsx/output.csv
│  └─ English relation data/
│         forest_foods.csv/_relations.csv/_class.csv/_species.csv/_family.csv/_genus.csv/_order.csv/_phylum.csv/_contains.csv/_belongs_to.csv
│         English_dataset.txt
│
├─ data_process/
│      clean_eval.py
│      data_process.py
│      Get_data.py
│      Get_data_2.py
│
└─ NER/
    │  bert_bilstm_crf.py
    │  bert_bilstm_softmax.py
    │  bert_softmax.py
    │  main.py
    │  requirements.txt
    │  utils.py
    │
    ├─ data2label/
    │      data2labeltrain.txt
    │      data2labeltest.txt
    │      processedown.py
    │      营养成分BIO标注结果.txt / v2 / v3
    │
    └─ results/

```

---

## File descriptions and formats

---

### `dataset/`

Contains the final entity-relationship triples in both Chinese and English formats.

```
Example (English version):

entity1, relation, entity2
"Elderberry", "contains", "Vitamin C"
"Oak", "belongs_to", "Fagaceae"


Example (Chinese version):

实体1, 关系, 实体2
"接骨木", "含有", "维生素C"
"橡树", "属于", "壳斗科"
```

---

### `data_process/`

This directory manages the end-to-end Chinese data pipeline: Web Crawling, Taxonomic Structuring, Fuzzy Matching/Cleaning, and Quality Evaluation. It consolidates data from two Chinese sources into a single, structured, and cleaned knowledge base for subsequent NER and KG construction.

* `Get_data.py` / `Get_data_2.py` – Web crawling and data collection.

* `data_process.py` – Cleaning and preprocessing of raw data.

* `clean_eval.py` – Evaluation or verification of processed data.

Output files from this folder are used as input for the NER module.

---

### `NER/`

This module handles Named Entity Recognition (NER) model training and inference.

* `bert_bilstm_crf.py`, `bert_softmax.py`, `bert_cnn_crf.py`: Model architectures combining BERT with CRF or Softmax classifiers.

* `main.py`: Entry point for training or evaluating the NER models.

* `utils.py`: Data loading and helper functions.

Outputs are stored in:

`NER/data2label/` – Processed and labeled training/testing data.

`NER/results/` – Model predictions and evaluation results.

Dependencies are listed in `NER/requirements.txt`.

---

## Environment and Dependencies

We recommend using Python 3.8+, 3.10 for the origin version used in this project.

```
bash
cd NER
pip install -r requirements.txt
```

If you plan to build and query the knowledge graph, install Neo4j (v4.x or v5.x) and ensure it’s running locally.

---

## Quick start step-by-step 

This section explains how to reproduce the whole project, mainly on data preprocessing, NER model training (BERT-based), and result verification based on structure above.

### 0) Clone the repository

```bash
git clone https://github.com/dadadaray/FTAND/code_dataset_KG.git
cd code_dataset_KG
```

### 1) Prepare the Python environment

We recommend using Python 3.8+, 3.10 for the origin version used in this project. Dependencies are listed in NER/requirements.txt.

```bash
python -m venv .venv
source .venv/bin/activate      # or `.venv\Scripts\activate` on Windows
pip install -r NER/requirements.txt
```

### 2) Validate data cleaning and quantify improvements(Optional)
`Raw_dataset.txt` have been prossessed and outputed as `Prosessed_dataset.txt`, if you're interested in quantify index we're using, run:
```
head -n 10 examples/demo/demo_triples.csv
```
You can compare these metrics before/after running data_process.py to quantitatively assess cleaning performance.

### 3) Run BERT-based NER experiments

NER models (BERT + CRF/Softmax/CNN) are defined in the `NER/` directory.

First, choose the model type for training and evaluation from `bert_softmax`,`bert_bilstm_softmax`,`bert_bilstm_crf`. 

After chosing the right model, set the path for trainset and validset you are using. 

For this project, trainset and validset can be accessed through`NER/data2label/data2labeltrain.txt` and `NER/data2label/data2labeltest.txt`, which contains a dataset labeled in our benchmark. 

Then use the batch size and epochs using for this training, run:

```bash
cd NER
python main.py \
--model_type bert_bilstm_softmax \
--trainset NER/data2label/data2labeltrain.txt \
--validset NER/data2label/data2labeltest.txt \
--batch_size 8 \
--n_epochs 30
```
If you already have checkponts,run:

```bash
python main.py \
--model_type bert_bilstm_softmax \
--validset NER/data2label/dev.jsonl \
--checkpoint checkpoints/bert_bilstm_softmax_final.pt \
--only_eval \
--logdir NER/results
```
Example expected output:
```
从 resultss/best_model.pt 加载模型参数...
模型初始化完成。

开始评估 2000 条样本...
逐字预测结果已保存到: NER/results/eval_output_detailed.txt
B-FOOD - 精确度: 0.910000, 召回率: 0.905000, F1: 0.907487, 数量: 400
I-FOOD - 精确度: 0.880000, 召回率: 0.865000, F1: 0.872439, 数量: 600
B-NUT - 精确度: 0.820000, 召回率: 0.815000, F1: 0.817498, 数量: 300
I-NUT - 精确度: 0.800000, 召回率: 0.795000, F1: 0.797492, 数量: 500

Macro 总体 - 精确度: 0.852500, 召回率: 0.845000, F1: 0.848731
Micro 总体 - 精确度: 0.923400, 召回率: 0.915500, F1: 0.919434
```
### 4) Build a simple Neo4j knowledge graph (optional)

You can optionally visualize and query relationships using Neo4j.

* Launch Neo4j Desktop or Server

* Copy the English dataset (e.g. forest_foods_relations.csv) to Neo4j’s import folder

* In Neo4j Browser, execute:

```bash
LOAD CSV WITH HEADERS FROM 'file:///forest_foods_relations.csv' AS row
MERGE (a:Entity {name: row.entity1})
MERGE (b:Entity {name: row.entity2})
MERGE (a)-[:RELATION {type: row.relation}]->(b);
```

You can then test queries like:

```bash
MATCH (f:Entity)-[r:RELATION]->(n:Entity)
WHERE r.type='contains'
RETURN f.name AS Food, n.name AS Nutrient LIMIT 20;
```
---
# Conclusion

The FoodKG project establishes a transparent and extensible framework for integrating forest food knowledge into structured graph form.
Through automated data collection, BERT-based entity recognition, and bilingual relationship extraction, it bridges the gap between raw textual data and structured semantic representations.

### We envision FoodKG as a foundation for:

Nutritional and ecological research, connecting food components to taxonomic and biochemical information.

Cross-lingual natural language processing, promoting the alignment of Chinese–English entity knowledge.

Knowledge graph reasoning, enabling intelligent applications such as food–nutrition recommendation and species discovery.

### Future work will extend FoodKG by:

Incorporating more fine-grained nutrient relationships (e.g., compounds, health effects).

Expanding coverage to more ecological and geographical sources.

Providing an API and interactive query interface for real-time graph exploration.

**FoodKG is open for academic collaboration and welcomes contributions from researchers interested in knowledge graphs, food science, or computational linguistics.**
