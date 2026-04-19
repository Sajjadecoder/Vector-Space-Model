# Vector Space Model - Information Retrieval System

## Overview

This project implements a complete **Vector Space Model (VSM)** for information retrieval using TF-IDF ranking and cosine similarity. It provides both command-line and GUI interfaces for querying a corpus of documents (speeches). The system preprocesses documents, builds search indexes, and retrieves relevant documents ranked by similarity to user queries.

---

## System Pipeline

```
Raw Documents  ──Preprocess──▶  Processed Docs
                 (lowercase,       (vocabulary)
                 tokenize,
                 stopwords,
                 lemmatize)
                                    │
                                    ▼
                            Build Indexes
                            ├─ Inverted Index
                            ├─ TF
                            ├─ IDF
                            └─ TF-IDF Vectors
                                    │
                                    ▼
                            ┌──────────────────┐
                            │  Loaded Indexes  │
                            │   (in memory)    │
                            └────────┬─────────┘
                                     │
    User Query──Preprocess──┐       │
    (same pipeline)        │       │
                           ▼       ▼
                    ┌─────────────────────────┐
                    │ Query Processing       │
                    ├─ Query Vector (TF-IDF) │
                    ├─ Candidate Retrieval   │
                    ├─ Cosine Similarity     │
                    └────────┬────────────────┘
                             │
                             ▼
                    ┌──────────────────────┐
                    │ Filter & Rank        │
                    │ (Alpha threshold)    │
                    └────────┬─────────────┘
                             │
                             ▼
                    ┌──────────────────────┐
                    │ Ranked Results       │
                    │ Display to User      │
                    └──────────────────────┘
```

---

## Technical Components

### 1. **Text Preprocessing Pipeline**
- **Case Folding:** Convert all text to lowercase
- **Tokenization:** Extract words using regex pattern `\b[a-z0-9]+\b`
- **Stopword Removal:** Filter common English stopwords
- **Lemmatization:** Use spaCy NLP library for linguistic lemmatization

**File:** `src/preprocess.py`

### 2. **Indexing System**

#### Vocabulary Generation
- Extract all unique terms from the corpus
- Sort alphabetically for consistent ordering
- Store in JSON format

#### Inverted Index
- Maps each term to list of document IDs containing it
- Used for candidate document retrieval

#### Term Frequency (TF)
- Count occurrence of each term in each document
- Format: `{doc_id: {term: count, ...}}`

#### Inverse Document Frequency (IDF)
- Measures importance of term across corpus
- Formula: `log(N / df)` where N = total docs, df = doc frequency
- Higher IDF = rarer term = more discriminative

#### TF-IDF Weights
- Combines TF and IDF: `weight = tf * idf`
- Creates dense vectors for each document
- Used for similarity calculation

**Files:** `src/vocabulary_generation.py`, `src/inverted_index.py`, `src/tf.py`, `src/idf.py`, `src/tf-idf.py`

### 3. **Query Processing & Ranking**

#### Query Vector Construction
- Preprocess query with same pipeline as documents
- Build TF-IDF vector for query terms
- Use IDF values from corpus

#### Similarity Scoring
- **Cosine Similarity:** Calculate angle between query and document vectors
- Formula: `similarity = (Q·D) / (||Q|| * ||D||)`
- Range: 0 to 1 (1 = perfect match)

#### Result Filtering & Ranking
- Apply alpha threshold (default: 0.005) to filter low scores
- Sort results by similarity score (highest first)
- Return top results with ranks

**File:** `src/query_processor.py`

### 4. **User Interfaces**

#### Command-Line Interface
- Interactive query mode
- Batch query processing from file
- Command-line arguments for configuration

#### GUI Interface (tkinter)
- Modern, formal design with professional color scheme
- Real-time search with instant results
- Visual display of corpus statistics
- Keyboard shortcuts (Ctrl+Enter for search)
- Scrollable results table with full document details

**File:** `main.py`

---

## Installation & Setup

### Prerequisites
- Python 3.7+
- pip (Python package manager)

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Download spaCy Language Model
```bash
python -m spacy download en_core_web_sm
```

### Step 3: Verify Directory Structure
Ensure the following directories exist:
- `data/speeches/` - Contains document corpus
- `indexes/` - Will store generated indexes

---

## Usage

### Launch GUI Application
```bash
python main.py
```

**GUI Features:**
- Enter queries in the search box
- Press `Ctrl+Enter` or click "Search" button
- View ranked results with similarity scores
- Clear button to reset query and results

### Command-Line Interface
```bash
python src/query_processor.py
```

**Interactive Mode:**
- Type queries at the prompt
- Type 'exit' to quit
- Results displayed in formatted table

**Batch Processing:**
```bash
python src/query_processor.py --queries data/queries.txt
```

**Custom Parameters:**
```bash
python src/query_processor.py --alpha 0.01 --topk 5
```

---

## Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `alpha` | 0.005 | Similarity threshold for result filtering |
| `topk` | 10 | Maximum number of results to display |
| `queries` | None | Path to batch query file |

---

## Key Algorithms

### Cosine Similarity
```
similarity(q, d) = (∑ q_i × d_i) / (√∑ q_i² × √∑ d_i²)

Where:
  q = query vector (TF-IDF weights)
  d = document vector (TF-IDF weights)
  Result range: [0, 1]
```

### TF-IDF Weight
```
tfidf(t, d) = tf(t, d) × idf(t)

Where:
  tf(t, d) = count of term t in document d
  idf(t) = log(N / df(t))
  N = total number of documents
  df(t) = number of documents containing term t
```

---

## Performance Characteristics

- **Corpus Size:** 50 documents
- **Vocabulary Size:** ~5,000+ unique terms (after preprocessing)
- **Query Processing Time:** < 100ms per query
- **Memory Usage:** ~50MB (indexes loaded in memory)

---

## Example Output

```
============================================================
  Query : machine learning applications
============================================================
  Rank   Score      Doc ID   Filename
  ─────────────────────────────────────────────────────────
  1      0.847521   5        speech_5.txt
  2      0.756234   12       speech_12.txt
  3      0.693412   18       speech_18.txt
  4      0.614523   3        speech_3.txt
  5      0.587234   28       speech_28.txt
```

---

## Dependencies

- **tkinter:** GUI framework (built-in with Python)
- **spacy:** Natural Language Processing for lemmatization
- **en_core_web_sm:** spaCy English language model
- **re, os, json, math:** Python standard library modules

See `requirements.txt` for complete dependency list.

---

## Technical Details

### Preprocessing
The text preprocessing pipeline applies the following transformations in order:
1. Lowercase conversion (case folding)
2. Tokenization with regex matching
3. Stopword removal (common English words)
4. Lemmatization (word root extraction)

### Vector Space Model
The system implements the standard Vector Space Model using:
- **Indexing:** TF-IDF weight calculation
- **Query Representation:** Query vector construction
- **Similarity:** Cosine similarity between vectors
- **Ranking:** Results sorted by similarity score

### Scalability
For larger corpora, consider:
- Implementing positional indexing for phrase queries
- Using approximate nearest neighbor search
- Implementing query caching
- Distributing indexes across multiple machines

---

## Author
Created as an Information Retrieval course assignment.

## License
Academic Project

---

## Notes

- All indexes are pre-computed and stored as JSON files
- Queries are processed in real-time without re-indexing
- The system is designed for demonstration and educational purposes
- For production use, consider using specialized IR frameworks (Elasticsearch, Solr, etc.)

