import os
import json
import logging
import time
import re
from typing import Dict, List, Tuple, Any, Optional, Union
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from openai import OpenAI
from datasets import load_dataset

# Import from our existing modules
from search import SearchEngine, load_corpus_from_beir
from evaluation import Evaluator, load_qrels
from query_expansion import QueryExpander

from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("llm_reranker")


class LLMReranker:
    """
    Class for reranking search results using LLM and cross-encoder models.
    Implements LLM-based reranking with explanations and cross-encoder reranking for comparison.
    """

    def __init__(self,
                 cache_dir: str = "cache",
                 reranked_dir: str = "cache/reranked",
                 model: str = "gpt-4o-mini"):
        """
        Initialize the LLM reranker.

        Args:
            cache_dir: Base directory for caching
            reranked_dir: Directory for caching reranked results
            model: OpenAI model to use for reranking
        """
        self.cache_dir = cache_dir
        self.reranked_dir = reranked_dir
        self.model = model

        # Create cache directories
        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(reranked_dir, exist_ok=True)

        # Initialize OpenAI client
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.warning("OPENAI_API_KEY not found in environment variables!")
        self.client = OpenAI(api_key=api_key)

        # Define the reranking prompt (Chain-of-Thought style)
        # Notice the double curly braces to escape them in the format string
        self.reranking_prompt = """
        Below is a search query and a list of {num_docs} retrieved documents (titles + snippets).

        1. Think through the query and what relevant means for the query.
        2. Go through the documents and rank these documents in order of relevance.
        3. For each document, provide a 2-line explanation of why it is ranked there.
        4. Return a JSON list: [{{"doc_id": "...", "rank": 1, "explanation": "..."}}, ...]

        Query: {query}

        Documents:
        {docs}
        """

    def _call_openai(self, prompt: str) -> str:
        """
        Call the OpenAI API with the given prompt.

        Args:
            prompt: Prompt to send to the OpenAI API

        Returns:
            Response from the OpenAI API
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,  # Lower temperature for more consistent ranking
                max_tokens=1024,  # Increased for longer explanations
                response_format={"type": "json_object"}  # Ensure JSON format
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {e}")
            return ""

    def _parse_reranking_response(self, response: str, doc_ids: List[str]) -> List[Tuple[str, float, str]]:
        """
        Parse the response from OpenAI API to extract reranking information.

        Args:
            response: Response from the OpenAI API
            doc_ids: Original list of document IDs

        Returns:
            List of (doc_id, score, explanation) tuples
        """
        try:
            # Parse the JSON response
            response_data = json.loads(response)

            # Check for different possible formats of the response
            if isinstance(response_data, list):
                reranked_docs = response_data
            elif isinstance(response_data, dict) and "rankings" in response_data:
                reranked_docs = response_data["rankings"]
            elif isinstance(response_data, dict) and "documents" in response_data:
                reranked_docs = response_data["documents"]
            elif isinstance(response_data, dict) and len(response_data) > 0:
                # Look for any list in the response
                for key, value in response_data.items():
                    if isinstance(value, list) and len(value) > 0:
                        reranked_docs = value
                        break
                else:
                    # If no list found, create a fallback
                    logger.warning(f"Could not find ranking list in response")
                    reranked_docs = []
            else:
                logger.warning(f"Unexpected response format: {response}")
                reranked_docs = []

            # Verify reranked_docs is a list
            if not isinstance(reranked_docs, list):
                logger.warning(f"Parsed result is not a list: {reranked_docs}")
                reranked_docs = []

            # Create result list
            result = []
            original_doc_ids = set(doc_ids)
            found_doc_ids = set()

            # Process each document in the reranked list
            for i, item in enumerate(reranked_docs):
                if not isinstance(item, dict):
                    continue

                # Extract doc_id, ensuring it's one of the original doc_ids
                doc_id = item.get("doc_id")
                if doc_id is None:
                    continue

                if doc_id not in original_doc_ids:
                    # Try to match with the original document IDs
                    closest_match = None
                    for orig_id in original_doc_ids:
                        if orig_id in doc_id or doc_id in orig_id:
                            closest_match = orig_id
                            break

                    if closest_match:
                        doc_id = closest_match
                    else:
                        continue

                found_doc_ids.add(doc_id)

                # Get rank and explanation
                rank = item.get("rank", i + 1)  # Use position as fallback if rank not provided
                explanation = item.get("explanation", "No explanation provided")

                # Normalize score (1.0 for highest rank, decreasing as rank increases)
                score = 1.0 / rank if rank > 0 else 0.0

                result.append((doc_id, score, explanation))

            # Sort by score (descending)
            result.sort(key=lambda x: x[1], reverse=True)

            # Add any missing documents with low scores
            missing_doc_ids = original_doc_ids - found_doc_ids
            for doc_id in missing_doc_ids:
                result.append((doc_id, 0.0, "Not ranked by LLM"))

            return result

        except Exception as e:
            logger.error(f"Error parsing reranking response: {e}")
            logger.error(f"Response: {response}")

            # Return original doc_ids with normalized scores as fallback
            return [(doc_id, 1.0 - (i / len(doc_ids)), "Failed to parse ranking")
                    for i, doc_id in enumerate(doc_ids)]

    def rerank_documents_with_llm(self,
                                  query: str,
                                  doc_ids: List[str],
                                  corpus: Dict[str, Dict[str, str]],
                                  force_rerank: bool = False) -> List[Tuple[str, float, str]]:
        """
        Rerank documents using an LLM (OpenAI) and provide explanations.

        Args:
            query: Original query
            doc_ids: List of document IDs to rerank
            corpus: Dictionary mapping document IDs to document content
            force_rerank: Whether to force reranking (ignore cache)

        Returns:
            List of (doc_id, score, explanation) tuples
        """
        # Prepare for caching
        query_hash = query.replace(' ', '_')
        cache_file = os.path.join(self.reranked_dir, f"llm_reranked_{query_hash}.json")

        # Check cache
        if os.path.exists(cache_file) and not force_rerank:
            logger.info(f"Loading LLM reranked results from cache for '{query}'")
            with open(cache_file, 'r') as f:
                return json.load(f)

        logger.info(f"Reranking documents with LLM for query: '{query}'")

        # Prepare documents for the prompt
        docs_str = ""
        for i, doc_id in enumerate(doc_ids):
            doc = corpus.get(doc_id, {})
            title = doc.get("title", "No title")
            text = doc.get("text", "No text")

            # Truncate text to a reasonable length to avoid exceeding token limits
            if len(text) > 400:
                text = text[:397] + "..."

            docs_str += f"{i + 1}. [Document ID: {doc_id}]\nTitle: {title}\nSnippet: {text}\n\n"

        # Prepare the prompt
        prompt = self.reranking_prompt.format(
            query=query,
            docs=docs_str,
            num_docs=len(doc_ids)
        )

        # Call OpenAI API
        response = self._call_openai(prompt)

        # Parse response
        reranked_results = self._parse_reranking_response(response, doc_ids)

        # Cache results
        with open(cache_file, 'w') as f:
            json.dump(reranked_results, f)

        return reranked_results

    def format_results_for_evaluation(self,
                                      results: List[Tuple[str, float, str]]) -> Tuple[
        List[Tuple[str, float]], Dict[str, str]]:
        """
        Format results for evaluation and separate explanations.

        Args:
            results: List of (doc_id, score, explanation) tuples

        Returns:
            Tuple of (evaluation_results, explanations)
            evaluation_results: List of (doc_id, score) tuples
            explanations: Dictionary mapping doc_ids to explanations
        """
        evaluation_results = [(doc_id, score) for doc_id, score, _ in results]
        explanations = {doc_id: explanation for doc_id, _, explanation in results}

        return evaluation_results, explanations

    def save_reranked_results_with_explanations(self,
                                                query: str,
                                                results: List[Tuple[str, float, str]],
                                                corpus: Dict[str, Dict[str, str]],
                                                method: str = "llm"):
        """
        Save reranked results with explanations in a readable format.

        Args:
            query: Original query
            results: List of (doc_id, score, explanation) tuples
            corpus: Dictionary mapping document IDs to document content
            method: Reranking method ("llm" or "cross_encoder")
        """
        # Create results directory
        results_dir = os.path.join("results", "reranking")
        os.makedirs(results_dir, exist_ok=True)

        # Create a readable file
        query_hash = query.replace(' ', '_')
        output_file = os.path.join(results_dir, f"{method}_reranked_{query_hash}.txt")

        with open(output_file, 'w') as f:
            f.write(f"Query: {query}\n\n")
            f.write(f"Reranking Method: {method}\n\n")
            f.write(f"Reranked Results:\n")
            f.write("=" * 80 + "\n\n")

            for i, (doc_id, score, explanation) in enumerate(results):
                doc = corpus.get(doc_id, {})
                title = doc.get("title", "No title")
                text = doc.get("text", "No text")

                f.write(f"Rank {i + 1}: {doc_id} (Score: {score:.4f})\n")
                f.write(f"Title: {title}\n")
                f.write(f"Explanation: {explanation}\n")
                f.write(f"Text Snippet: {text[:200]}...\n")
                f.write("-" * 80 + "\n\n")

        logger.info(f"Saved {method} reranked results to {output_file}")


def main():
    """
    Main function to run LLM reranking experiments.
    Implements Stage 2 of the project:
    - Load best performing expanded queries from Stage 1
    - Retrieve initial candidate documents
    - Rerank using LLM with explanations
    - Compare with cross-encoder reranking
    - Evaluate and visualize results
    """
    # Log start time
    start_time = time.time()

    # Configuration options
    force_rerank = False  # Whether to force reranking (ignore cache)
    use_subset = False  # Whether to use a subset of queries
    max_queries = 5  # Maximum number of queries to process if use_subset is True
    top_k_retrieve = 20  # Number of documents to retrieve for reranking

    # Evaluation metrics configuration
    p_k = 20  # Precision@k value
    r_k = 500  # Recall@k value
    ndcg_k = 20  # NDCG@k value

    # Initialize LLM reranker
    llm_reranker = LLMReranker()

    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY environment variable not set!")
        logger.error("Please set the OPENAI_API_KEY environment variable to your OpenAI API key.")
        return

    # Load TREC-COVID dataset
    logger.info("Loading TREC-COVID dataset")
    corpus_dataset = load_dataset("BeIR/trec-covid", "corpus")["corpus"]
    queries_dataset = load_dataset("BeIR/trec-covid", "queries")["queries"]
    qrels_dataset = load_dataset("BeIR/trec-covid-qrels", split="test")

    # Format corpus
    corpus = load_corpus_from_beir(corpus_dataset)

    # Initialize the search engine
    search_engine = SearchEngine()

    # Create BM25 index
    search_engine.create_bm25_index(corpus, force_reindex=False)

    # Create semantic index
    search_engine.create_semantic_index(
        corpus,
        model_name="all-mpnet-base-v2",
        force_reindex=False
    )

    # Load cross-encoder model for comparison
    search_engine.load_cross_encoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")

    # Get qrels for evaluation
    qrels = load_qrels(qrels_dataset)

    # Initialize evaluator
    evaluator = Evaluator(qrels, results_dir="results/reranking")

    # Initialize query expander to access cached zero-shot expansions
    query_expander = QueryExpander()

    # Create a mapping from numeric query ID to query text
    query_id_to_text = {}
    for item in queries_dataset:
        query_id = int(item["_id"])
        query_text = item["text"]
        query_id_to_text[query_id] = query_text

    logger.info(f"Found {len(query_id_to_text)} queries with text")

    # Define methods to evaluate
    methods = {
        "initial_retrieval": {},  # Initial retrieval results (no reranking)
        "cross_encoder": {},  # Cross-encoder reranking
        "llm_reranking": {}  # LLM-based reranking with explanations
    }

    # Define metrics for evaluation
    metrics = [f"p@{p_k}", f"r@{r_k}", f"ndcg@{ndcg_k}"]

    # Process each query in qrels
    logger.info("Processing queries with relevance judgments")

    # Get query IDs for processing
    query_ids = list(qrels.keys())

    # Use subset if requested
    if use_subset and max_queries < len(query_ids):
        logger.info(f"Using subset of {max_queries} queries (out of {len(query_ids)})")
        query_ids = query_ids[:max_queries]

    # Create a directory to store reranked document explanations
    os.makedirs("results/reranking", exist_ok=True)

    for query_id in tqdm(query_ids, desc="Evaluating queries"):
        if query_id not in query_id_to_text:
            logger.warning(f"Query ID {query_id} not found in queries dataset. Skipping.")
            continue

        query_text = query_id_to_text[query_id]

        try:
            # Step 1: Use zero-shot expanded queries to retrieve initial candidates
            # Load cached zero-shot expanded queries
            expanded_queries = query_expander.expand_query_zero_shot(
                query_text,
                force_expand=False  # Use cached expanded queries
            )

            # Retrieve top-k documents using the expanded queries
            retrieval_results = query_expander.perform_hybrid_retrieval(
                search_engine,
                query_text,
                expanded_queries,
                top_k=max(top_k_retrieve * 2, r_k),  # Get enough for recall calculation
                is_weighted=True,
                force_retrieve=False  # Use cached retrieval results
            )

            # Extract document IDs for reranking (we only rerank the top ones)
            doc_ids = [doc_id for doc_id, _ in retrieval_results[:top_k_retrieve]]

            # Save initial retrieval results for evaluation
            methods["initial_retrieval"][query_id] = [doc_id for doc_id, _ in retrieval_results]

            # Step 2: Rerank using cross-encoder
            cross_encoder_results = search_engine.rerank_cross_encoder(
                query_text,
                doc_ids,
                top_k=top_k_retrieve
            )

            # Get reranked doc IDs
            cross_encoder_doc_ids = [doc_id for doc_id, _ in cross_encoder_results]

            # Get remaining docs from initial retrieval (to have enough for recall calculation)
            remaining_docs = [doc_id for doc_id, _ in retrieval_results
                              if doc_id not in cross_encoder_doc_ids]

            # Save cross-encoder results for evaluation
            methods["cross_encoder"][query_id] = cross_encoder_doc_ids + remaining_docs

            # Step 3: Rerank using LLM with explanations
            llm_reranked_results = llm_reranker.rerank_documents_with_llm(
                query_text,
                doc_ids,
                corpus,
                force_rerank=force_rerank
            )

            # Format results for evaluation
            llm_eval_results, explanations = llm_reranker.format_results_for_evaluation(llm_reranked_results)

            # Get reranked doc IDs
            llm_doc_ids = [doc_id for doc_id, _ in llm_eval_results]

            # Get remaining docs from initial retrieval (to have enough for recall calculation)
            remaining_docs = [doc_id for doc_id, _ in retrieval_results
                              if doc_id not in llm_doc_ids]

            # Save LLM reranking results for evaluation
            methods["llm_reranking"][query_id] = llm_doc_ids + remaining_docs

            # Save reranked results with explanations for manual evaluation
            llm_reranker.save_reranked_results_with_explanations(
                query_text,
                llm_reranked_results,
                corpus,
                method="llm"
            )

        except Exception as e:
            logger.error(f"Error processing query {query_id}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            continue

    # Evaluate and store results for each method
    logger.info("Evaluating results")

    # Dictionary to store results for visualization
    all_results = {}

    for method_name, run_results in methods.items():
        logger.info(f"Evaluating {method_name}")
        logger.info(f"{method_name} has results for {len(run_results)} queries")

        # Sample check
        if run_results:
            sample_query_id = next(iter(run_results))
            sample_docs = run_results[sample_query_id][:5]
            logger.info(f"Sample docs for query {sample_query_id}: {sample_docs}")

        results = evaluator.evaluate_run(run_results, metrics=metrics)
        evaluator.save_results(results, method_name, "reranking")

        # Store mean metrics for visualization
        all_results[method_name] = {metric: results[metric]["mean"] for metric in metrics}

        # Log mean metrics
        for metric in metrics:
            logger.info(f"{method_name} {metric}: {results[metric]['mean']:.4f}")

    # Compare all methods
    logger.info("Comparing all methods")
    comparison = evaluator.compare_runs("reranking", list(methods.keys()), metrics=metrics)

    # Log comparison results
    logger.info("Comparison results:")
    for run, metrics_values in comparison.items():
        metrics_str = ", ".join(f"{k}: {v:.4f}" for k, v in metrics_values.items())
        logger.info(f"{run}: {metrics_str}")

    # Create visualizations
    create_visualizations(all_results, metrics)

    # Log total execution time
    execution_time = time.time() - start_time
    logger.info(f"Total execution time: {execution_time:.2f} seconds")

    logger.info("Evaluation complete. Results saved to results/reranking/")


def create_visualizations(results: Dict[str, Dict[str, float]], metrics: List[str]):
    """
    Create visualizations for the evaluation results.

    Args:
        results: Dictionary mapping methods to dictionaries of metrics
        metrics: List of metrics
    """
    # Create directory for visualizations
    os.makedirs("results/reranking", exist_ok=True)

    # Create bar chart for each metric
    for metric in metrics:
        plt.figure(figsize=(10, 6))

        methods = list(results.keys())
        values = [results[method][metric] for method in methods]

        plt.bar(methods, values)
        plt.title(f"Reranking Performance Comparison - {metric}")
        plt.xlabel("Method")
        plt.ylabel(f"{metric}")
        plt.xticks(rotation=45)

        # Add value labels
        for i, v in enumerate(values):
            plt.text(i, v + 0.01, f"{v:.3f}", ha="center")

        plt.tight_layout()
        plt.savefig(f"results/reranking/{metric}_comparison.png")
        plt.close()

    # Create combined visualization
    plt.figure(figsize=(12, 8))

    methods = list(results.keys())
    x = np.arange(len(methods))
    width = 0.25 if len(metrics) <= 3 else 0.2

    offset = width * (1 - len(metrics)) / 2
    for i, metric in enumerate(metrics):
        values = [results[method][metric] for method in methods]
        plt.bar(x + offset + i * width, values, width, label=metric)

    plt.title("Reranking Performance Comparison Across Metrics")
    plt.xlabel("Method")
    plt.ylabel("Score")
    plt.xticks(x, methods, rotation=45)
    plt.legend()

    plt.tight_layout()
    plt.savefig("results/reranking/combined_comparison.png")
    plt.close()


if __name__ == "__main__":
    main()