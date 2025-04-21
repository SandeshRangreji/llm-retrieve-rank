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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("search")


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

        Args:
            cache_dir: Directory for caching indices and results
            device: Device to use for model inference ('cuda', 'mps', 'cpu', or None for auto-detection)
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

        # Initialize indices
        self.corpus = None
        self.doc_ids = None
        self.bm25_index = None
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.semantic_model = None
        self.semantic_embeddings = None
        self.cross_encoder = None

    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text for indexing/searching.

        Args:
            text: Input text

        Returns:
            Preprocessed text
        """
        # Simple preprocessing - could be expanded based on needs
        text = text.lower()
        return text

    def create_bm25_index(self,
                          corpus: Dict[str, Dict[str, str]],
                          field: str = "text",
                          force_reindex: bool = False) -> None:
        """
        Create a BM25 index for the corpus.

        Args:
            corpus: Dictionary mapping document IDs to document content
                   Format: {doc_id: {"title": title, "text": text, ...}, ...}
            field: Field to index
            force_reindex: Whether to force reindexing (ignore cache)
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

        # Extract document texts
        doc_ids = []
        texts = []

        for doc_id, doc in tqdm(corpus.items(), desc="Extracting documents"):
            doc_ids.append(doc_id)
            text = self.preprocess_text(doc[field])
            texts.append(text)

        # Tokenize texts
        tokenized_texts = [text.split() for text in texts]

        # Create BM25 index
        self.bm25_index = BM25Okapi(tokenized_texts)
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
            text = self.preprocess_text(doc[field])
            texts.append(text)

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
        query = self.preprocess_text(query)
        tokenized_query = query.split()

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

        # Preprocess query
        query = self.preprocess_text(query)

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
            device=self.device
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
                      method: str = "rrf",
                      alpha: float = 0.5,
                      k_factor: int = 60) -> List[Tuple[str, float]]:
        """
        Search the corpus using a hybrid of BM25 and semantic search.

        Args:
            query: Search query
            top_k: Number of top results to return
            method: Hybridization method ('rrf' for Reciprocal Rank Fusion, 'score' for weighted score sum)
            alpha: Weight for BM25 in score sum (1-alpha for semantic)
            k_factor: k factor for RRF

        Returns:
            List of (doc_id, score) tuples
        """
        # Get BM25 results
        bm25_results = self.search_bm25(query, top_k=top_k * 2)

        # Get semantic results
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
            # Normalized Score Sum

            # Normalize BM25 scores (min-max)
            if bm25_results:
                min_bm25 = min(score for _, score in bm25_results)
                max_bm25 = max(score for _, score in bm25_results)
                bm25_range = max_bm25 - min_bm25
                if bm25_range == 0:
                    bm25_range = 1.0  # Avoid division by zero
            else:
                min_bm25, bm25_range = 0, 1.0

            # Normalize semantic scores (min-max)
            if semantic_results:
                min_sem = min(score for _, score in semantic_results)
                max_sem = max(score for _, score in semantic_results)
                sem_range = max_sem - min_sem
                if sem_range == 0:
                    sem_range = 1.0  # Avoid division by zero
            else:
                min_sem, sem_range = 0, 1.0

            # Combine scores
            combined_scores = defaultdict(float)

            # Add BM25 scores
            for doc_id, score in bm25_results:
                normalized_score = (score - min_bm25) / bm25_range
                combined_scores[doc_id] += alpha * normalized_score

            # Add semantic scores
            for doc_id, score in semantic_results:
                normalized_score = (score - min_sem) / sem_range
                combined_scores[doc_id] += (1 - alpha) * normalized_score

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
    Runs keyword, semantic, hybrid+score, and hybrid+RRF searches and evaluates each.
    """
    try:
        from datasets import load_dataset
        from evaluation import Evaluator, load_qrels

        # Load TREC-COVID dataset
        logger.info("Loading TREC-COVID dataset")
        corpus_dataset = load_dataset("BeIR/trec-covid", "corpus")["corpus"]
        queries_dataset = load_dataset("BeIR/trec-covid", "queries")["queries"]
        qrels_dataset = load_dataset("BeIR/trec-covid-qrels", split="test")

        # Format corpus and qrels
        corpus = load_corpus_from_beir(corpus_dataset)
        qrels = load_qrels(qrels_dataset)

        # Initialize search engine
        search_engine = SearchEngine()

        # Create BM25 index
        logger.info("Creating BM25 index")
        search_engine.create_bm25_index(corpus)

        # Create semantic index
        logger.info("Creating semantic index")
        search_engine.create_semantic_index(corpus)

        # Initialize evaluator
        evaluator = Evaluator(qrels, results_dir="results/search_comparison")

        # Prepare results dictionaries for each method
        methods = {
            "bm25": {},
            "semantic": {},
            "hybrid_score": {},
            "hybrid_rrf": {}
        }

        # Process each query
        logger.info(f"Processing {len(queries_dataset)} queries")
        for query_item in tqdm(queries_dataset, desc="Evaluating queries"):
            query_id = query_item["_id"]
            query_text = query_item["text"]

            # BM25 search
            bm25_results = search_engine.search_bm25(query_text, top_k=500)
            methods["bm25"][query_id] = [doc_id for doc_id, _ in bm25_results]

            # Semantic search
            semantic_results = search_engine.search_semantic(query_text, top_k=500)
            methods["semantic"][query_id] = [doc_id for doc_id, _ in semantic_results]

            # Hybrid search with score combination
            hybrid_score_results = search_engine.search_hybrid(query_text, top_k=500, method="score")
            methods["hybrid_score"][query_id] = [doc_id for doc_id, _ in hybrid_score_results]

            # Hybrid search with RRF
            hybrid_rrf_results = search_engine.search_hybrid(query_text, top_k=500, method="rrf")
            methods["hybrid_rrf"][query_id] = [doc_id for doc_id, _ in hybrid_rrf_results]

        # Evaluate and store results for each method
        logger.info("Evaluating results")
        metrics = ["p@20", "r@500", "ndcg@20"]

        for method_name, run_results in methods.items():
            logger.info(f"Evaluating {method_name}")
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

        logger.info("Evaluation complete. Results saved to results/search_comparison/")

    except ImportError:
        logger.error("Could not import the 'datasets' or 'evaluation' modules. Please install them to run this evaluation.")
        logger.info("Skipping full evaluation. Run a simple demo with a sample corpus instead.")

        # Create a sample corpus for testing
        sample_corpus = {
            "doc1": {"title": "COVID-19 Overview", "text": "COVID-19 is a disease caused by the SARS-CoV-2 virus."},
            "doc2": {"title": "Symptoms of COVID", "text": "Common symptoms include fever, cough, and fatigue."},
            "doc3": {"title": "Treatment Options", "text": "Treatment focuses on managing symptoms and supporting vital functions."},
            "doc4": {"title": "Vaccines", "text": "Several vaccines have been developed to prevent COVID-19."},
            "doc5": {"title": "Safety Measures", "text": "Masks, social distancing, and hand hygiene help prevent the spread."}
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
