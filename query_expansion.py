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

from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("query_expansion")


class QueryExpander:
    """
    Class for expanding queries using different LLM prompting strategies.
    Implements various prompting techniques for query expansion and caches results.
    """

    def __init__(self,
                 cache_dir: str = "cache",
                 expanded_queries_dir: str = "cache/expanded_queries",
                 retrieved_results_dir: str = "cache/retrieved_results",
                 model: str = "gpt-4o-mini"):
        """
        Initialize the query expander.

        Args:
            cache_dir: Base directory for caching
            expanded_queries_dir: Directory for caching expanded queries
            retrieved_results_dir: Directory for caching retrieval results
            model: OpenAI model to use for query expansion
        """
        self.cache_dir = cache_dir
        self.expanded_queries_dir = expanded_queries_dir
        self.retrieved_results_dir = retrieved_results_dir
        self.model = model

        # Create cache directories
        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(expanded_queries_dir, exist_ok=True)
        os.makedirs(retrieved_results_dir, exist_ok=True)

        # Initialize OpenAI client
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.warning("OPENAI_API_KEY not found in environment variables!")
        self.client = OpenAI(api_key=api_key)

        # Define prompt templates with consistent JSON output format
        self.zero_shot_prompt = """
        Expand the following query to improve search relevance. 
        Generate 5 alternative queries and assign a relevance weight from 1-10 for each.
        Return results as JSON: [{{"query": "...", "weight": 9}}, ...].

        Query: {query}
        """

        self.few_shot_prompt = """
        I'll show you examples of query expansion with weights, then I want you to expand a new query.

        Original query: COVID-19 transmission
        Expanded queries with weights (1-10):
        [
          {{"query": "How does COVID-19 spread from person to person", "weight": 9}},
          {{"query": "SARS-CoV-2 transmission methods", "weight": 8}},
          {{"query": "COVID-19 airborne transmission evidence", "weight": 10}},
          {{"query": "Coronavirus droplet transmission research", "weight": 7}},
          {{"query": "COVID-19 surface transmission studies", "weight": 6}}
        ]

        Original query: mRNA vaccine technology
        Expanded queries with weights (1-10):
        [
          {{"query": "How do mRNA vaccines like Pfizer and Moderna work", "weight": 10}},
          {{"query": "mRNA vaccine development history and mechanism", "weight": 9}},
          {{"query": "Differences between mRNA vaccines and traditional vaccines", "weight": 8}},
          {{"query": "mRNA vaccine technology breakthrough explained", "weight": 7}},
          {{"query": "mRNA lipid nanoparticle delivery system", "weight": 6}}
        ]

        Now, expand the following query with 5 alternatives and assign a relevance weight from 1-10 for each.
        Return results as JSON: [{{"query": "...", "weight": 9}}, ...].

        Query: {query}
        """

        self.cot_prompt = """
        First think through adjacent topics related to this query. Put together diverse topics that would help retrieve relevant information.
        Then, formulate 5 different query alternatives that approach the topic from different angles.
        For each alternative query, assign a relevance weight from 1-10 based on how relevant and useful you think it would be.

        Return your final result as JSON: [{{"query": "...", "weight": 9}}, ...].

        Query: {query}
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
                temperature=0.7,
                max_tokens=400
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {e}")
            return ""

    def _parse_response_to_weighted_queries(self, response: str) -> List[Tuple[str, float]]:
        """
        Parse the response from OpenAI API to extract weighted queries.

        Args:
            response: Response from the OpenAI API

        Returns:
            List of (expanded_query, weight) tuples
        """
        try:
            # Find the JSON part within the response
            json_start = response.find('[')
            json_end = response.rfind(']') + 1

            if json_start != -1 and json_end != -1:
                json_str = response[json_start:json_end]
                expanded_queries = json.loads(json_str)

                # Convert to list of tuples
                result = [(item["query"], item["weight"]) for item in expanded_queries]
            else:
                # Fallback: parse manually
                logger.warning(f"Could not find JSON in response, trying manual parsing")
                lines = response.split('\n')
                result = []

                for line in lines:
                    if ':' in line and ('weight' in line.lower() or 'relevance' in line.lower()):
                        parts = line.split(':', 1)
                        if len(parts) == 2:
                            try:
                                query_text = parts[0].strip().strip('"\'').strip()
                                weight_text = parts[1].strip()
                                # Extract number from weight text
                                weight_match = re.search(r'\d+', weight_text)
                                if weight_match:
                                    weight = float(weight_match.group())
                                    result.append((query_text, weight))
                            except Exception:
                                pass

                if not result:
                    logger.warning(f"Failed to parse expanded queries with weights. Using original query.")
                    result = [("original query", 10.0)]
        except Exception as e:
            logger.error(f"Error parsing weighted expanded queries: {e}")
            logger.error(f"Response: {response}")
            result = [("original query", 10.0)]  # Fallback to original query

        return result

    def expand_query_zero_shot(self, query: str, force_expand: bool = False) -> List[Tuple[str, float]]:
        """
        Expand a query using the zero-shot prompting strategy.

        Args:
            query: Original query
            force_expand: Whether to force expansion (ignore cache)

        Returns:
            List of (expanded_query, weight) tuples
        """
        cache_file = os.path.join(self.expanded_queries_dir, f"zero_shot_{query.replace(' ', '_')}.json")

        # Check cache
        if os.path.exists(cache_file) and not force_expand:
            logger.info(f"Loading expanded queries from cache for '{query}' (zero-shot)")
            with open(cache_file, 'r') as f:
                return json.load(f)

        # Expand query
        logger.info(f"Expanding query using zero-shot prompting: '{query}'")
        prompt = self.zero_shot_prompt.format(query=query)
        response = self._call_openai(prompt)

        # Parse response to get weighted queries
        expanded_queries = self._parse_response_to_weighted_queries(response)

        # Cache results
        with open(cache_file, 'w') as f:
            json.dump(expanded_queries, f)

        return expanded_queries

    def expand_query_few_shot(self, query: str, force_expand: bool = False) -> List[Tuple[str, float]]:
        """
        Expand a query using the few-shot prompting strategy.

        Args:
            query: Original query
            force_expand: Whether to force expansion (ignore cache)

        Returns:
            List of (expanded_query, weight) tuples
        """
        cache_file = os.path.join(self.expanded_queries_dir, f"few_shot_{query.replace(' ', '_')}.json")

        # Check cache
        if os.path.exists(cache_file) and not force_expand:
            logger.info(f"Loading expanded queries from cache for '{query}' (few-shot)")
            with open(cache_file, 'r') as f:
                return json.load(f)

        # Expand query
        logger.info(f"Expanding query using few-shot prompting: '{query}'")
        prompt = self.few_shot_prompt.format(query=query)
        response = self._call_openai(prompt)

        # Parse response to get weighted queries
        expanded_queries = self._parse_response_to_weighted_queries(response)

        # Cache results
        with open(cache_file, 'w') as f:
            json.dump(expanded_queries, f)

        return expanded_queries

    def expand_query_cot(self, query: str, force_expand: bool = False) -> List[Tuple[str, float]]:
        """
        Expand a query using the chain-of-thought prompting strategy.

        Args:
            query: Original query
            force_expand: Whether to force expansion (ignore cache)

        Returns:
            List of (expanded_query, weight) tuples
        """
        cache_file = os.path.join(self.expanded_queries_dir, f"cot_{query.replace(' ', '_')}.json")

        # Check cache
        if os.path.exists(cache_file) and not force_expand:
            logger.info(f"Loading expanded queries from cache for '{query}' (chain-of-thought)")
            with open(cache_file, 'r') as f:
                return json.load(f)

        # Expand query
        logger.info(f"Expanding query using chain-of-thought prompting: '{query}'")
        prompt = self.cot_prompt.format(query=query)
        response = self._call_openai(prompt)

        # Parse response to get weighted queries
        expanded_queries = self._parse_response_to_weighted_queries(response)

        # Cache results
        with open(cache_file, 'w') as f:
            json.dump(expanded_queries, f)

        return expanded_queries

    def perform_hybrid_retrieval(self,
                                 search_engine: SearchEngine,
                                 query: str,
                                 expanded_queries: List[Union[str, Tuple[str, float]]],
                                 top_k: int = 100,
                                 is_weighted: bool = False,
                                 force_retrieve: bool = False) -> List[Tuple[str, float]]:
        """
        Perform hybrid retrieval using expanded queries.

        Args:
            search_engine: SearchEngine instance
            query: Original query
            expanded_queries: List of expanded queries or (query, weight) tuples
            top_k: Number of documents to retrieve
            is_weighted: Whether expanded_queries contains weights
            force_retrieve: Whether to force retrieval (ignore cache)

        Returns:
            List of (doc_id, score) tuples
        """
        # Prepare for caching
        query_hash = query.replace(' ', '_')
        cache_type = "weighted" if is_weighted else "unweighted"
        cache_file = os.path.join(self.retrieved_results_dir, f"{cache_type}_{query_hash}.json")

        # Check cache
        if os.path.exists(cache_file) and not force_retrieve:
            logger.info(f"Loading retrieval results from cache for '{query}'")
            with open(cache_file, 'r') as f:
                return json.load(f)

        logger.info(f"Performing hybrid retrieval for '{query}' with {len(expanded_queries)} expanded queries")

        # Initialize score accumulator
        all_scores = defaultdict(float)

        # Process each expanded query
        if is_weighted:
            for expanded_query, weight in expanded_queries:
                # Normalize weight to [0, 1]
                normalized_weight = weight / 10.0

                # Retrieve documents
                hybrid_results = search_engine.search_hybrid(
                    expanded_query,
                    top_k=top_k,
                    method="normalize"
                )

                # Accumulate scores with weights
                for doc_id, score in hybrid_results:
                    all_scores[doc_id] += score * normalized_weight
        else:
            for expanded_query in expanded_queries:
                # Retrieve documents
                hybrid_results = search_engine.search_hybrid(
                    expanded_query,
                    top_k=top_k,
                    method="normalize"
                )

                # Accumulate scores with equal weights
                for doc_id, score in hybrid_results:
                    all_scores[doc_id] += score / len(expanded_queries)

        # Sort by accumulated score
        results = [(doc_id, score) for doc_id, score in all_scores.items()]
        results.sort(key=lambda x: x[1], reverse=True)

        # Truncate to top_k
        results = results[:top_k]

        # Cache results
        with open(cache_file, 'w') as f:
            json.dump(results, f)

        return results


def main():
    """
    Main function to run query expansion experiments.
    Implements Stage 1 of the project:
    - Load TREC-COVID dataset
    - Expand queries using different prompting strategies
    - Perform hybrid retrieval with expanded queries
    - Evaluate retrieval results
    """
    # Log start time
    start_time = time.time()

    # Configuration options
    force_expand = False  # Whether to force query expansion (ignore cache)
    force_retrieve = False  # Whether to force retrieval (ignore cache)
    use_subset = False  # Whether to use a subset of queries
    max_queries = 5  # Maximum number of queries to process if use_subset is True

    # Initialize query expander
    query_expander = QueryExpander()

    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY environment variable not set!")
        logger.error("Please set the OPENAI_API_KEY environment variable to your OpenAI API key.")
        return

    # Load TREC-COVID dataset as instructed
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

    # Get qrels for evaluation
    qrels = load_qrels(qrels_dataset)

    # Initialize evaluator
    evaluator = Evaluator(qrels, results_dir="results/query_expansion")

    # Create a mapping from numeric query ID to query text
    query_id_to_text = {}
    for item in queries_dataset:
        query_id = int(item["_id"])
        query_text = item["text"]
        query_id_to_text[query_id] = query_text

    logger.info(f"Found {len(query_id_to_text)} queries with text")

    # Define methods to evaluate
    methods = {
        "original": {},  # No expansion
        "zero_shot": {},
        "few_shot": {},
        "cot": {}
    }

    # Define metrics for evaluation
    metrics = ["p@20", "r@500", "ndcg@20"]

    # Process each query in qrels
    logger.info("Processing queries with relevance judgments")

    # Get query IDs for processing
    query_ids = list(qrels.keys())

    # Use subset if requested
    if use_subset and max_queries < len(query_ids):
        logger.info(f"Using subset of {max_queries} queries (out of {len(query_ids)})")
        query_ids = query_ids[:max_queries]

    for query_id in tqdm(query_ids, desc="Evaluating queries"):
        if query_id not in query_id_to_text:
            logger.warning(f"Query ID {query_id} not found in queries dataset. Skipping.")
            continue

        query_text = query_id_to_text[query_id]

        # Original query (baseline)
        baseline_results = search_engine.search_hybrid(
            query_text,
            top_k=1000,
            method="normalize"
        )
        methods["original"][query_id] = [doc_id for doc_id, _ in baseline_results]

        # Zero-shot expansion with weights
        expanded_queries = query_expander.expand_query_zero_shot(
            query_text,
            force_expand=force_expand
        )
        zero_shot_results = query_expander.perform_hybrid_retrieval(
            search_engine,
            query_text,
            expanded_queries,
            top_k=1000,
            is_weighted=True,  # Use weighted retrieval
            force_retrieve=force_retrieve
        )
        methods["zero_shot"][query_id] = [doc_id for doc_id, _ in zero_shot_results]

        # Few-shot expansion with weights
        expanded_queries = query_expander.expand_query_few_shot(
            query_text,
            force_expand=force_expand
        )
        few_shot_results = query_expander.perform_hybrid_retrieval(
            search_engine,
            query_text,
            expanded_queries,
            top_k=1000,
            is_weighted=True,  # Use weighted retrieval
            force_retrieve=force_retrieve
        )
        methods["few_shot"][query_id] = [doc_id for doc_id, _ in few_shot_results]

        # Chain-of-thought expansion with weights
        expanded_queries = query_expander.expand_query_cot(
            query_text,
            force_expand=force_expand
        )
        cot_results = query_expander.perform_hybrid_retrieval(
            search_engine,
            query_text,
            expanded_queries,
            top_k=1000,
            is_weighted=True,  # Use weighted retrieval
            force_retrieve=force_retrieve
        )
        methods["cot"][query_id] = [doc_id for doc_id, _ in cot_results]

    # Rest of the function remains the same...

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
        evaluator.save_results(results, method_name, "query_expansion")

        # Store mean metrics for visualization
        all_results[method_name] = {metric: results[metric]["mean"] for metric in metrics}

        # Log mean metrics
        for metric in metrics:
            logger.info(f"{method_name} {metric}: {results[metric]['mean']:.4f}")

    # Compare all methods
    logger.info("Comparing all methods")
    comparison = evaluator.compare_runs("query_expansion", list(methods.keys()), metrics=metrics)

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

    logger.info("Evaluation complete. Results saved to results/query_expansion/")


def create_visualizations(results: Dict[str, Dict[str, float]], metrics: List[str]):
    """
    Create visualizations for the evaluation results.

    Args:
        results: Dictionary mapping methods to dictionaries of metrics
        metrics: List of metrics
    """
    # Create directory for visualizations
    os.makedirs("results/query_expansion", exist_ok=True)

    # Create bar chart for each metric
    for metric in metrics:
        plt.figure(figsize=(10, 6))

        methods = list(results.keys())
        values = [results[method][metric] for method in methods]

        plt.bar(methods, values)
        plt.title(f"Performance comparison - {metric}")
        plt.xlabel("Method")
        plt.ylabel(f"{metric}")
        plt.xticks(rotation=45)

        # Add value labels
        for i, v in enumerate(values):
            plt.text(i, v + 0.01, f"{v:.3f}", ha="center")

        plt.tight_layout()
        plt.savefig(f"results/query_expansion/{metric}_comparison.png")
        plt.close()

    # Create combined visualization
    plt.figure(figsize=(12, 8))

    methods = list(results.keys())
    x = np.arange(len(methods))
    width = 0.25

    for i, metric in enumerate(metrics):
        values = [results[method][metric] for method in methods]
        plt.bar(x + (i - 1) * width, values, width, label=metric)

    plt.title("Performance comparison across metrics")
    plt.xlabel("Method")
    plt.ylabel("Score")
    plt.xticks(x, methods, rotation=45)
    plt.legend()

    plt.tight_layout()
    plt.savefig("results/query_expansion/combined_comparison.png")
    plt.close()


if __name__ == "__main__":
    main()