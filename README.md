# Querying the Mind of a Model: Prompt-Driven Expansion and Re-ranking for Information Retrieval

This project implements a three-stage information retrieval system that explores query expansion, re-ranking, and web-based QA using large language models.

## 🔧 Environment Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up environment variables:**
   Create a `.env` file in the project root:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   TAVILY_API_KEY=your_tavily_api_key_here
   ```

## 📊 Stage Overview

### Stage 1: Query Expansion (`query_expansion.py`)
- Expands queries using different LLM prompting strategies
- Evaluates retrieval performance with expanded queries
- **Dependencies:** `search.py`, `evaluation.py`
- **Outputs:** Cached expanded queries, retrieval results, evaluation metrics

### Stage 2: LLM Re-ranking (`llm_reranker.py`)
- Re-ranks documents using LLM with explanations
- Compares against cross-encoder re-ranking
- **Dependencies:** `search.py`, `evaluation.py`, `query_expansion.py`
- **Uses:** Best performing expansion method from Stage 1
- **Outputs:** Re-ranked results with explanations, comparative evaluations

### Stage 3: Web Search + QA (`llm_search.py`)
- Performs web search using Tavily API
- Generates answers with explanations and follow-up suggestions
- **Dependencies:** All previous modules (for reusable components)
- **Outputs:** Formatted QA results, visualizations

## 🔨 File Descriptions and Parameters

### 1. `evaluation.py`
**Purpose:** Provides evaluation framework for all stages.

**Key Classes:**
- `Evaluator`: Computes P@k, R@k, NDCG@k metrics

**Configurable Parameters in `main()`:**
- Sample qrels and run results for testing
- No major configuration needed - primarily a utility module

### 2. `search.py`
**Purpose:** Core search engine with keyword, semantic, and hybrid search.

**Key Classes:**
- `SearchEngine`: BM25, semantic (sentence-transformers), hybrid search

**Configurable Parameters in `main()`:**
```python
# Dataset and corpus settings
corpus_dataset = "BeIR/trec-covid"  # Can change to other BeIR datasets
queries_dataset = "BeIR/trec-covid"

# Search parameters
search_methods = ["bm25", "semantic", "hybrid_normalize"]
semantic_model = "all-mpnet-base-v2"  # Can use other sentence-transformers models
cross_encoder_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Evaluation metrics
metrics = ["p@20", "r@1000", "ndcg@20"]  # Customizable precision/recall/ndcg values

# Index creation
force_reindex = False  # Set to True to rebuild indices
```

### 3. `query_expansion.py`
**Purpose:** Stage 1 - LLM-based query expansion with different prompting strategies.

**Configurable Parameters in `main()`:**
```python
# Expansion control
force_expand = False      # Force new expansions (ignore cache)
force_retrieve = False    # Force new retrievals (ignore cache)

# Query processing
use_subset = False        # Use subset of queries for testing
max_queries = 5          # Number of queries if using subset

# LLM settings
openai_model = "gpt-4o-mini"  # OpenAI model for expansion

# Search parameters
top_k_retrieve = 1000    # Number of documents to retrieve
search_method = "normalize"  # Hybrid search combination method

# Evaluation metrics
metrics = ["p@20", "r@500", "ndcg@20"]
```

**Interdependencies:**
- Uses `SearchEngine` from `search.py` for retrieval
- Uses `Evaluator` from `evaluation.py` for metrics
- **Outputs used by Stage 2:** Cached expanded queries (especially zero-shot)

### 4. `llm_reranker.py`
**Purpose:** Stage 2 - LLM-based re-ranking with explanations.

**Configurable Parameters in `main()`:**
```python
# Re-ranking control
force_rerank = False      # Force new re-rankings (ignore cache)

# Query processing
use_subset = False        # Use subset of queries for testing
max_queries = 5          # Number of queries if using subset

# Retrieval parameters
top_k_retrieve = 20      # Number of docs to retrieve for re-ranking

# Evaluation metrics configuration
p_k = 20                 # Precision@k value
r_k = 500               # Recall@k value  
ndcg_k = 20             # NDCG@k value

# LLM settings
openai_model = "gpt-4o-mini"  # OpenAI model for re-ranking
temperature = 0.2        # Lower for consistent ranking

# Cross-encoder model
cross_encoder_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"
```

**Interdependencies:**
- **Uses from Stage 1:** Best performing expansion method (zero-shot cached results)
- Uses `SearchEngine` for initial retrieval and cross-encoder re-ranking
- Uses `QueryExpander` to load cached expanded queries
- **Outputs:** Re-ranked results with explanations for manual evaluation

### 5. `llm_search.py`
**Purpose:** Stage 3 - Web search + LLM QA system.

**Configurable Parameters in `main()`:**
```python
# Execution mode
mode = "visualize"        # Options: "interactive", "batch", "single", "visualize"

# Query settings
single_query = "Should I wear a mask in April 2020?"  # For single mode
suggested_queries = [     # Predefined complex queries
    "Should I wear a mask in April 2020?",
    "Can COVID spread through surfaces?",
    # ... more queries
]

# Cache control
force_search = False      # Force new web searches (ignore cache)
force_answer = False      # Force new LLM answers (ignore cache)

# Output settings
output_dir = "results/web_qa"  # Directory for evaluation results

# API settings
openai_model = "gpt-4o-mini"   # OpenAI model for QA
search_depth = "basic"         # Tavily search depth ("basic" or "advanced")
max_results = 7               # Max web search results to use

# LLM parameters
temperature = 0.3             # Temperature for answer generation
max_tokens = 1024            # Max tokens for LLM response
```

**Interdependencies:**
- Can import and use components from all previous modules
- Uses `TavilyClient` for web search
- **Outputs:** QA results, visualizations, interactive CLI interface

## 🔄 Data Flow Between Files

### Stage 1 → Stage 2
- **Cache Location:** `cache/expanded_queries/`
- **File Pattern:** `zero_shot_*.json`, `few_shot_*.json`, `cot_*.json`
- **Content:** List of `[(expanded_query, weight), ...]` tuples
- **Usage:** Stage 2 loads zero-shot expanded queries to retrieve initial candidates

### Stage 2 → Manual Evaluation
- **Cache Location:** `cache/reranked/`
- **File Pattern:** `llm_reranked_*.json`
- **Content:** List of `[(doc_id, score, explanation), ...]` tuples
- **Output:** `results/reranking/*_reranked_*.txt` for manual evaluation

### Cross-Stage Caching
All stages use intelligent caching:
- **Search indices:** `cache/bm25_index.pkl`, `cache/semantic_index_*.pkl`
- **Retrieval results:** `cache/retrieved_results/`
- **Web searches:** `cache/tavily_results/`

## 🚀 Running the Project

### Sequential Execution (Recommended)
```bash
# Stage 1: Query expansion evaluation
python query_expansion.py

# Stage 2: Re-ranking with best method from Stage 1
python llm_reranker.py

# Stage 3: Web search QA (interactive mode)
python llm_search.py
```

### Individual Components
```bash
# Test search engine only
python search.py

# Run evaluation framework
python evaluation.py

# Interactive web QA
python llm_search.py  # Set mode="interactive"
```

## 📈 Evaluation and Results

### Metrics Used
- **Precision@20:** Proportion of relevant docs in top 20
- **Recall@500/1000:** Proportion of relevant docs found
- **NDCG@20:** Normalized discounted cumulative gain

### Output Locations
- **Numerical Results:** `results/{stage_name}/*.json`
- **Visualizations:** `results/{stage_name}/comparison.png`
- **Manual Evaluation:** `results/reranking/*.txt` (with explanations)
- **Web QA:** `results/web_qa/*.md` and visualizations

## ⚙️ Customization Tips

1. **Change Dataset:** Modify corpus loading in each file to use different BeIR datasets
2. **Adjust Models:** 
   - Semantic: Change `semantic_model` in SearchEngine
   - LLM: Modify `openai_model` parameters
   - Cross-encoder: Update model name in search.py
3. **Experiment with Prompts:** Edit prompt templates in query_expansion.py and llm_reranker.py
4. **Evaluation Metrics:** Adjust k values for P@k, R@k, NDCG@k as needed

## 🔧 Troubleshooting

- **GPU/MPS Usage:** Automatic device detection in SearchEngine
- **API Rate Limits:** Built-in caching reduces API calls
- **Memory Issues:** Reduce `max_queries` or `top_k_retrieve` values
- **Missing Dependencies:** Check imports and install required packages

This modular design allows you to run individual stages, experiment with different parameters, and build upon previous results while maintaining full reproducibility through caching.