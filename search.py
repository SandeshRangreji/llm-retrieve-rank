import os
import json
import pickle
import logging
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
from collections import defaultdict
import torch
from torch import Tensor
from tqdm import tqdm

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder, util
from rank_bm25 import BM25Okapi
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import nltk
import re
import string

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("search")


# Download NLTK resources
def download_nltk_resources():
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('punkt')
        nltk.download('stopwords')


class SearchEngine:
    """
    Search engine class that supports keyword, semantic, and hybrid search methods.
    Also supports cross-encoder reranking.
    """

    def __init__(self,
                 cache_dir: str = "cache",
                 device: Optional[str] = None):
        """
        Initialize the search engine.
        """
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

        # Set device
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        logger.info(f"Using device: {self.device}")

        # Download NLTK resources for text preprocessing
        download_nltk_resources()

        # Initialize stemmer and stopwords for keyword search
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))

        # Initialize indices
        self.corpus = None
        self.doc_ids = None
        self.bm25_index = None
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.semantic_model = None
        self.semantic_embeddings = None
        self.cross_encoder = None

    def preprocess_text(self, text: str, for_keyword: bool = False):
        """
        Preprocess text for indexing/searching.

        Args:
            text: Input text
            for_keyword: Whether to apply more rigorous preprocessing for keyword search

        Returns:
            Preprocessed tokens (list) for keyword search
        """
        if for_keyword:
            # Match the reference implementation exactly
            text = text.lower()
            text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
            tokens = text.split()  # Split on whitespace
            filtered_tokens = [self.stemmer.stem(w) for w in tokens if w not in self.stop_words]
            return filtered_tokens

        return text.lower()

    def create_bm25_index(self,
                          corpus: Dict[str, Dict[str, str]],
                          field: str = "text",
                          force_reindex: bool = False) -> None:
        """
        Create a BM25 index for the corpus.
        """
        cache_path = os.path.join(self.cache_dir, "bm25_index.pkl")

        # Check if index exists in cache
        if os.path.exists(cache_path) and not force_reindex:
            logger.info("Loading BM25 index from cache")
            with open(cache_path, "rb") as f:
                cached_data = pickle.load(f)
                self.bm25_index = cached_data["index"]
                self.doc_ids = cached_data["doc_ids"]
                self.corpus = cached_data.get("corpus", corpus)
            return

        logger.info("Creating BM25 index")

        # Store corpus
        self.corpus = corpus

        # Extract document texts - exactly match the reference implementation
        doc_ids = []
        texts = []

        for doc_id, doc in tqdm(corpus.items(), desc="Extracting documents"):
            doc_ids.append(doc_id)  # Keep original ID format

            # Combine title and text exact same way as reference implementation
            if "title" in doc and "text" in doc:
                combined_text = doc["title"] + "\n\n" + doc["text"]  # Note: literal "/n/n", not newlines
            else:
                combined_text = doc.get(field, "")

            texts.append(combined_text)  # Store raw text first

        # Tokenize corpus after collecting all texts (same as reference)
        tokenized_corpus = [self.preprocess_text(text, for_keyword=True) for text in texts]

        # Create BM25 index with tuned parameters (b=0.5 as in the reference code)
        self.bm25_index = BM25Okapi(tokenized_corpus, b=0.5)
        self.doc_ids = doc_ids

        # Cache the index
        with open(cache_path, "wb") as f:
            pickle.dump({
                "index": self.bm25_index,
                "doc_ids": self.doc_ids,
                "corpus": self.corpus
            }, f)

        logger.info(f"BM25 index created and cached at {cache_path}")

    def create_tfidf_index(self,
                           corpus: Dict[str, Dict[str, str]],
                           field: str = "text",
                           force_reindex: bool = False) -> None:
        """
        Create a TF-IDF index for the corpus.

        Args:
            corpus: Dictionary mapping document IDs to document content
            field: Field to index
            force_reindex: Whether to force reindexing (ignore cache)
        """
        cache_path = os.path.join(self.cache_dir, "tfidf_index.pkl")

        # Check if index exists in cache
        if os.path.exists(cache_path) and not force_reindex:
            logger.info("Loading TF-IDF index from cache")
            with open(cache_path, "rb") as f:
                cached_data = pickle.load(f)
                self.tfidf_vectorizer = cached_data["vectorizer"]
                self.tfidf_matrix = cached_data["matrix"]
                self.doc_ids = cached_data["doc_ids"]
                self.corpus = cached_data.get("corpus", corpus)
            return

        logger.info("Creating TF-IDF index")

        # Store corpus
        self.corpus = corpus

        # Extract document texts
        doc_ids = []
        texts = []

        for doc_id, doc in tqdm(corpus.items(), desc="Extracting documents"):
            doc_ids.append(doc_id)
            # Apply keyword-specific preprocessing for TF-IDF
            text = self.preprocess_text(doc[field], for_keyword=True)
            texts.append(" ".join(text))

        # Create TF-IDF index
        self.tfidf_vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
        self.doc_ids = doc_ids

        # Cache the index
        with open(cache_path, "wb") as f:
            pickle.dump({
                "vectorizer": self.tfidf_vectorizer,
                "matrix": self.tfidf_matrix,
                "doc_ids": self.doc_ids,
                "corpus": self.corpus
            }, f)

        logger.info(f"TF-IDF index created and cached at {cache_path}")

    def search_bm25(self,
                    query: str,
                    top_k: int = 100) -> List[Tuple[str, float]]:
        """
        Search the corpus using BM25.
        """
        if self.bm25_index is None or self.doc_ids is None:
            raise ValueError("BM25 index not created. Call create_bm25_index() first.")

        # Preprocess and tokenize query with keyword preprocessing
        tokenized_query = self.preprocess_text(query, for_keyword=True)

        # Get scores
        scores = self.bm25_index.get_scores(tokenized_query)

        # Get top-k results
        top_indices = np.argsort(scores)[::-1][:top_k]

        # Map indices to document IDs and scores
        results = [(self.doc_ids[idx], scores[idx]) for idx in top_indices if scores[idx] > 0]

        return results

    def create_semantic_index(self,
                              corpus: Dict[str, Dict[str, str]],
                              model_name: str = "all-mpnet-base-v2",
                              field: str = "text",
                              force_reindex: bool = False) -> None:
        """
        Create a semantic index for the corpus using a SentenceTransformer model.

        Args:
            corpus: Dictionary mapping document IDs to document content
            model_name: Name of the SentenceTransformer model
            field: Field to index
            force_reindex: Whether to force reindexing (ignore cache)
        """
        cache_path = os.path.join(self.cache_dir, f"semantic_index_{model_name.replace('/', '_')}.pkl")

        # Check if index exists in cache
        if os.path.exists(cache_path) and not force_reindex:
            logger.info(f"Loading semantic index for model {model_name} from cache")
            with open(cache_path, "rb") as f:
                cached_data = pickle.load(f)
                self.semantic_embeddings = cached_data["embeddings"]
                self.doc_ids = cached_data["doc_ids"]
                self.corpus = cached_data.get("corpus", corpus)

            # Load model
            logger.info(f"Loading semantic model: {model_name}")
            self.semantic_model = SentenceTransformer(model_name, device=self.device)

            return

        logger.info(f"Creating semantic index with model: {model_name}")

        # Store corpus
        self.corpus = corpus

        # Load model
        self.semantic_model = SentenceTransformer(model_name, device=self.device)

        # Extract document texts
        doc_ids = []
        texts = []

        for doc_id, doc in tqdm(corpus.items(), desc="Extracting documents"):
            doc_ids.append(doc_id)
            # For semantic search, we might want to include title and text
            if "title" in doc and field == "text":
                text = f"{doc.get('title', '')} {doc.get(field, '')}"
            else:
                text = doc.get(field, "")
            texts.append(text)

        # Generate embeddings
        logger.info("Generating document embeddings")
        self.semantic_embeddings = self.semantic_model.encode(
            texts,
            show_progress_bar=True,
            convert_to_tensor=True,
            device=self.device
        )
        self.doc_ids = doc_ids

        # Cache the index (convert tensor to numpy for pickling)
        with open(cache_path, "wb") as f:
            pickle.dump({
                "embeddings": self.semantic_embeddings.cpu().numpy() if isinstance(self.semantic_embeddings,
                                                                                   torch.Tensor) else self.semantic_embeddings,
                "doc_ids": self.doc_ids,
                "corpus": self.corpus
            }, f)

        logger.info(f"Semantic index created and cached at {cache_path}")

    def load_cross_encoder(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        """
        Load a cross-encoder model for re-ranking.

        Args:
            model_name: Name of the CrossEncoder model
        """
        logger.info(f"Loading cross-encoder model: {model_name}")
        self.cross_encoder = CrossEncoder(model_name, device=self.device)

    def search_bm25(self,
                    query: str,
                    top_k: int = 100) -> List[Tuple[str, float]]:
        """
        Search the corpus using BM25.

        Args:
            query: Search query
            top_k: Number of top results to return

        Returns:
            List of (doc_id, score) tuples
        """
        if self.bm25_index is None or self.doc_ids is None:
            raise ValueError("BM25 index not created. Call create_bm25_index() first.")

        # Preprocess and tokenize query
        tokenized_query = self.preprocess_text(query, for_keyword=True)

        # Get scores
        scores = self.bm25_index.get_scores(tokenized_query)

        # Get top-k results
        top_indices = np.argsort(scores)[::-1][:top_k]

        # Map indices to document IDs and scores
        results = [(self.doc_ids[idx], scores[idx]) for idx in top_indices if scores[idx] > 0]

        return results

    def search_tfidf(self,
                     query: str,
                     top_k: int = 100) -> List[Tuple[str, float]]:
        """
        Search the corpus using TF-IDF and cosine similarity.

        Args:
            query: Search query
            top_k: Number of top results to return

        Returns:
            List of (doc_id, score) tuples
        """
        if self.tfidf_vectorizer is None or self.tfidf_matrix is None or self.doc_ids is None:
            raise ValueError("TF-IDF index not created. Call create_tfidf_index() first.")

        # Preprocess query with keyword preprocessing
        query = " ".join(self.preprocess_text(query, for_keyword=True))

        # Transform query to TF-IDF space
        query_vector = self.tfidf_vectorizer.transform([query])

        # Calculate cosine similarities
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()

        # Get top-k results
        top_indices = np.argsort(similarities)[::-1][:top_k]

        # Map indices to document IDs and scores
        results = [(self.doc_ids[idx], similarities[idx]) for idx in top_indices if similarities[idx] > 0]

        return results

    def search_semantic(self,
                        query: str,
                        top_k: int = 100) -> List[Tuple[str, float]]:
        """
        Search the corpus using semantic embeddings and cosine similarity.

        Args:
            query: Search query
            top_k: Number of top results to return

        Returns:
            List of (doc_id, score) tuples
        """
        if (self.semantic_model is None or self.semantic_embeddings is None or
                self.doc_ids is None):
            raise ValueError("Semantic index not created. Call create_semantic_index() first.")

        # Encode query
        query_embedding = self.semantic_model.encode(
            query,
            convert_to_tensor=True,
            device=self.device,
            show_progress_bar=False
        )

        # Convert embeddings to tensor if they are numpy arrays
        if isinstance(self.semantic_embeddings, np.ndarray):
            self.semantic_embeddings = torch.tensor(
                self.semantic_embeddings,
                device=self.device
            )

        # Calculate cosine similarities
        similarities = util.cos_sim(query_embedding, self.semantic_embeddings)[0].cpu().numpy()

        # Get top-k results
        top_indices = np.argsort(similarities)[::-1][:top_k]

        # Map indices to document IDs and scores
        results = [(self.doc_ids[idx], similarities[idx]) for idx in top_indices]

        return results

    def search_hybrid(self,
                      query: str,
                      top_k: int = 100,
                      method: str = "normalize",
                      alpha: float = 0.5,
                      k_factor: int = 60) -> List[Tuple[str, float]]:
        """
        Search the corpus using a hybrid of BM25 and semantic search.

        Args:
            query: Search query
            top_k: Number of top results to return
            method: Hybridization method
                   'rrf' for Reciprocal Rank Fusion
                   'score' for weighted score sum
                   'normalize' for normalized score combination (reference implementation)
            alpha: Weight for BM25 in score sum (1-alpha for semantic)
            k_factor: k factor for RRF

        Returns:
            List of (doc_id, score) tuples
        """
        # Get BM25 results (get more to increase coverage for hybrid)
        bm25_results = self.search_bm25(query, top_k=top_k * 2)

        # Get semantic results (get more to increase coverage for hybrid)
        semantic_results = self.search_semantic(query, top_k=top_k * 2)

        if method == "rrf":
            # Reciprocal Rank Fusion
            fused_scores = defaultdict(float)

            # Process BM25 results
            for rank, (doc_id, _) in enumerate(bm25_results):
                fused_scores[doc_id] += 1.0 / (rank + k_factor)

            # Process semantic results
            for rank, (doc_id, _) in enumerate(semantic_results):
                fused_scores[doc_id] += 1.0 / (rank + k_factor)

            # Sort by score in descending order
            results = [(doc_id, score) for doc_id, score in fused_scores.items()]
            results.sort(key=lambda x: x[1], reverse=True)

            # Return top-k results
            return results[:top_k]

        elif method == "score":
            # Weighted Score Sum
            combined_scores = defaultdict(float)

            # Add BM25 scores
            for doc_id, score in bm25_results:
                combined_scores[doc_id] += alpha * score

            # Add semantic scores
            for doc_id, score in semantic_results:
                combined_scores[doc_id] += (1 - alpha) * score

            # Sort by score in descending order
            results = [(doc_id, score) for doc_id, score in combined_scores.items()]
            results.sort(key=lambda x: x[1], reverse=True)

            # Return top-k results
            return results[:top_k]

        elif method == "normalize":
            # Normalized Score Combination (implementation from reference code)
            # Get document IDs from both result sets
            bm25_doc_ids = [doc_id for doc_id, _ in bm25_results]
            semantic_doc_ids = [doc_id for doc_id, _ in semantic_results]

            # Create a combined set of document IDs (union)
            combined_doc_ids = set(bm25_doc_ids) | set(semantic_doc_ids)

            # Create dictionaries mapping doc_id to score
            bm25_scores_dict = {doc_id: score for doc_id, score in bm25_results}
            semantic_scores_dict = {doc_id: score for doc_id, score in semantic_results}

            # Find max scores for normalization
            max_bm25 = max(score for _, score in bm25_results) if bm25_results else 1.0
            max_semantic = max(score for _, score in semantic_results) if semantic_results else 1.0

            # Combine scores with normalization
            combined_scores = {}
            for doc_id in combined_doc_ids:
                # Get scores, default to 0 if not present in a result set
                bm25_score = bm25_scores_dict.get(doc_id, 0.0) / max_bm25
                semantic_score = semantic_scores_dict.get(doc_id, 0.0) / max_semantic

                # Simple sum of normalized scores
                combined_scores[doc_id] = bm25_score + semantic_score

            # Sort by score in descending order
            results = [(doc_id, score) for doc_id, score in combined_scores.items()]
            results.sort(key=lambda x: x[1], reverse=True)

            # Return top-k results
            return results[:top_k]

        else:
            raise ValueError(f"Unknown hybridization method: {method}")

    def rerank_cross_encoder(self,
                             query: str,
                             doc_ids: List[str],
                             top_k: int = 20) -> List[Tuple[str, float]]:
        """
        Re-rank the results using a cross-encoder model.

        Args:
            query: Search query
            doc_ids: List of document IDs to re-rank
            top_k: Number of top results to return

        Returns:
            List of (doc_id, score) tuples
        """
        if self.cross_encoder is None:
            raise ValueError("Cross-encoder not loaded. Call load_cross_encoder() first.")

        if self.corpus is None:
            raise ValueError("Corpus not loaded.")

        # Create input pairs for the cross-encoder
        pairs = []
        for doc_id in doc_ids:
            doc = self.corpus.get(doc_id, {})
            text = ""
            if "title" in doc:
                text += doc["title"] + " "
            if "text" in doc:
                text += doc["text"]

            pairs.append([query, text])

        # Get scores from cross-encoder
        scores = self.cross_encoder.predict(pairs)

        # Create a list of (doc_id, score) tuples and sort by score
        results = [(doc_id, score) for doc_id, score in zip(doc_ids, scores)]
        results.sort(key=lambda x: x[1], reverse=True)

        # Return top-k results
        return results[:top_k]

    def format_search_results(self,
                              results: List[Tuple[str, float]],
                              include_text: bool = False) -> Dict[str, Any]:
        """
        Format search results for output.

        Args:
            results: List of (doc_id, score) tuples
            include_text: Whether to include document text in the results

        Returns:
            Dictionary of formatted results
        """
        formatted_results = {
            "hits": len(results),
            "results": []
        }

        for doc_id, score in results:
            doc = self.corpus.get(doc_id, {})

            result = {
                "doc_id": doc_id,
                "score": float(score),  # Convert to float for serialization
                "title": doc.get("title", "")
            }

            if include_text:
                result["text"] = doc.get("text", "")

            formatted_results["results"].append(result)

        return formatted_results


def load_corpus_from_beir(dataset) -> Dict[str, Dict[str, str]]:
    """
    Load and format corpus from a BEIR dataset.

    Args:
        dataset: BEIR corpus dataset

    Returns:
        Dictionary mapping document IDs to document content
    """
    corpus = {}

    for item in dataset:
        # Use original ID format from dataset (don't convert to string)
        doc_id = item["_id"]
        title = item["title"]
        text = item["text"]

        corpus[doc_id] = {
            "title": title,
            "text": text
        }

    return corpus


def main():
    """
    Test and evaluate the SearchEngine class functionality with the TREC-COVID dataset.
    Runs keyword, semantic, hybrid searches and evaluates each.
    """
    try:
        from datasets import load_dataset
        from evaluation import Evaluator, load_qrels
        import time

        # Log start time
        start_time = time.time()

        # Load TREC-COVID dataset
        logger.info("Loading TREC-COVID dataset")
        corpus_dataset = load_dataset("BeIR/trec-covid", "corpus")["corpus"]
        queries_dataset = load_dataset("BeIR/trec-covid", "queries")["queries"]
        qrels_dataset = load_dataset("BeIR/trec-covid-qrels", split="test")

        # Log corpus size
        logger.info(f"Corpus size: {len(corpus_dataset)} documents")

        # Format corpus
        corpus = load_corpus_from_beir(corpus_dataset)

        # Initialize the search engine
        search_engine = SearchEngine()

        # Create BM25 index
        logger.info("Creating BM25 index")
        search_engine.create_bm25_index(corpus, force_reindex=False)
        logger.info(f"BM25 index size: {len(search_engine.doc_ids)} documents")

        # Create semantic index
        logger.info("Creating semantic index")
        search_engine.create_semantic_index(corpus, model_name="all-mpnet-base-v2", force_reindex=False)
        logger.info(f"Semantic index size: {len(search_engine.semantic_embeddings)} embeddings")

        # Get qrels for evaluation
        qrels = load_qrels(qrels_dataset)

        # Initialize evaluator
        evaluator = Evaluator(qrels, results_dir="results/search_comparison")

        # Create a mapping from numeric query ID to query text
        query_id_to_text = {}
        for item in queries_dataset:
            # Convert ID to int to match qrels IDs
            query_id = int(item["_id"])
            query_text = item["text"]
            query_id_to_text[query_id] = query_text

        logger.info(f"Found {len(query_id_to_text)} queries with text")

        # Prepare results dictionaries for each method
        methods = {
            "bm25": {},
            "semantic": {},
            "hybrid_normalize": {}
        }

        # Process each query in qrels
        num_evaluated = 0
        logger.info(f"Processing queries with relevance judgments")

        for query_id in tqdm(qrels.keys(), desc="Evaluating queries"):
            if query_id not in query_id_to_text:
                logger.warning(f"Query ID {query_id} not found in queries dataset. Skipping.")
                continue

            query_text = query_id_to_text[query_id]

            # BM25 search
            bm25_results = search_engine.search_bm25(query_text, top_k=1000)
            methods["bm25"][query_id] = [doc_id for doc_id, _ in bm25_results]

            # Semantic search
            semantic_results = search_engine.search_semantic(query_text, top_k=1000)
            methods["semantic"][query_id] = [doc_id for doc_id, _ in semantic_results]

            # Hybrid search with normalize
            hybrid_results = search_engine.search_hybrid(query_text, top_k=1000, method="normalize")
            methods["hybrid_normalize"][query_id] = [doc_id for doc_id, _ in hybrid_results]

            num_evaluated += 1

        logger.info(f"Evaluated {num_evaluated} queries")

        # Evaluate and store results for each method
        logger.info("Evaluating results")
        metrics = ["p@20", "r@1000", "ndcg@20"]

        for method_name, run_results in methods.items():
            logger.info(f"Evaluating {method_name}")
            logger.info(f"{method_name} has results for {len(run_results)} queries")

            # Sample check
            if run_results:
                sample_query_id = next(iter(run_results))
                sample_docs = run_results[sample_query_id][:5]
                logger.info(f"Sample docs for query {sample_query_id}: {sample_docs}")

                if sample_query_id in qrels:
                    sample_relevant = list(qrels[sample_query_id].keys())[:5]
                    logger.info(f"Sample relevant docs: {sample_relevant}")

                    # Check for any overlap
                    overlap = set(sample_docs).intersection(set(sample_relevant))
                    logger.info(f"Overlap in sample: {overlap}")

            results = evaluator.evaluate_run(run_results, metrics=metrics)
            evaluator.save_results(results, method_name, "search_types")

            # Log mean metrics
            for metric in metrics:
                logger.info(f"{method_name} {metric}: {results[metric]['mean']:.4f}")

        # Compare all methods
        logger.info("Comparing all methods")
        comparison = evaluator.compare_runs("search_types", list(methods.keys()), metrics=metrics)

        # Log comparison results
        logger.info("Comparison results:")
        for run, metrics_values in comparison.items():
            metrics_str = ", ".join(f"{k}: {v:.4f}" for k, v in metrics_values.items())
            logger.info(f"{run}: {metrics_str}")

        # Log total execution time
        execution_time = time.time() - start_time
        logger.info(f"Total execution time: {execution_time:.2f} seconds")

        logger.info("Evaluation complete. Results saved to results/search_comparison/")

    except ImportError:
        logger.error(
            "Could not import the 'datasets' or 'evaluation' modules. Please install them to run this evaluation.")
        logger.info("Skipping full evaluation. Run a simple demo with a sample corpus instead.")

        # Create a sample corpus for testing
        sample_corpus = {
            "doc1": {"title": "COVID-19 Overview", "text": "COVID-19 is a disease caused by the SARS-CoV-2 virus."},
            "doc2": {"title": "Symptoms of COVID", "text": "Common symptoms include fever, cough, and fatigue."},
            "doc3": {"title": "Treatment Options",
                     "text": "Treatment focuses on managing symptoms and supporting vital functions."},
            "doc4": {"title": "Vaccines", "text": "Several vaccines have been developed to prevent COVID-19."},
            "doc5": {"title": "Safety Measures",
                     "text": "Masks, social distancing, and hand hygiene help prevent the spread."}
        }

        # Initialize search engine
        search_engine = SearchEngine()

        # Create BM25 index
        search_engine.create_bm25_index(sample_corpus)

        # Test search
        test_query = "covid symptoms treatment"
        logger.info(f"Testing search with query: {test_query}")

        # BM25 search
        bm25_results = search_engine.search_bm25(test_query, top_k=5)
        logger.info(f"BM25 results: {len(bm25_results)} hits")
        for doc_id, score in bm25_results:
            logger.info(f"  {doc_id}: {score:.4f} - {sample_corpus[doc_id]['title']}")


if __name__ == "__main__":
    main()
