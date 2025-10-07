# Code Dataset Knowledge Graph
This project constructs a knowledge graph from code-related datasets, supporting both English and Chinese versions. The knowledge graph is built using Neo4j by importing entity-relationship triples.
code_dataset_KG/
├── data_process/          # Data crawling and processing
├── dataset/              # Entity-relationship triples dataset
│   ├── english/          # English version
│   └── chinese/          # Chinese version
└── NER/                  # Named Entity Recognition code

 data_process/
Contains scripts for:

Data crawling from various sources

Data cleaning and preprocessing

Data transformation and formatting

Preparation of raw data for NER processing

NER/
Named Entity Recognition module that includes:

Entity extraction from processed data

Entity classification and categorization

Relationship identification between entities

Model training and inference code

dataset/
Stores the final entity-relationship triples in format:

text
(entity1, relationship, entity2)
Available versions:

English version: Located in dataset/english/

Chinese version: Located in dataset/chinese/



Knowledge Graph Construction
Prerequisites
Neo4j database installed and running


Steps to Build Knowledge Graph
Prepare your Neo4j environment

Start Neo4j database

Ensure sufficient memory allocation for your dataset size

Import triples to Neo4j
