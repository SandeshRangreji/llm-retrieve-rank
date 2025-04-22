import numpy as np
import json
import os
import logging
from typing import Dict, List, Tuple, Any, Optional, Union
from collections import defaultdict
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("evaluation")


class Evaluator:
    """
    Evaluator class for information retrieval evaluation metrics.
    Provides functions to compute Precision@k, Recall@k, NDCG@k, and more.
    """

    def __init__(self, qrels: Dict[str, Dict[str, int]], results_dir: str = "results"):
        """
        Initialize the evaluator with ground truth relevance judgments.

        Args:
            qrels: Dictionary mapping query IDs to dictionaries of document IDs and relevance scores
                  Format: {query_id: {doc_id: relevance_score, ...}, ...}
            results_dir: Directory to save evaluation results
        """
        self.qrels = qrels
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)

    def precision_at_k(self,
                       query_id: str,
                       doc_ids: List[str],
                       k: int = 20) -> float:
        """
        Calculate Precision@k for a single query.

        Args:
            query_id: ID of the query
            doc_ids: List of retrieved document IDs (in ranked order)
            k: k value for Precision@k

        Returns:
            Precision@k value
        """
        if query_id not in self.qrels or not doc_ids:
            return 0.0

        relevant_docs = set(self.qrels[query_id].keys())
        retrieved_docs = doc_ids[:k]

        num_relevant_retrieved = len(set(retrieved_docs).intersection(relevant_docs))

        return num_relevant_retrieved / min(k, len(retrieved_docs))

    def recall_at_k(self,
                    query_id: str,
                    doc_ids: List[str],
                    k: int = 1000) -> float:
        """
        Calculate Recall@k for a single query.

        Args:
            query_id: ID of the query
            doc_ids: List of retrieved document IDs (in ranked order)
            k: k value for Recall@k

        Returns:
            Recall@k value
        """
        if query_id not in self.qrels or not doc_ids:
            return 0.0

        relevant_docs = set(self.qrels[query_id].keys())
        if not relevant_docs:
            return 0.0

        retrieved_docs = set(doc_ids[:k])

        num_relevant_retrieved = len(retrieved_docs.intersection(relevant_docs))

        return num_relevant_retrieved / len(relevant_docs)

    def dcg_at_k(self,
                 query_id: str,
                 doc_ids: List[str],
                 k: int = 20) -> float:
        """
        Calculate Discounted Cumulative Gain at k (DCG@k) for a single query.

        Args:
            query_id: ID of the query
            doc_ids: List of retrieved document IDs (in ranked order)
            k: k value for DCG@k

        Returns:
            DCG@k value
        """
        if query_id not in self.qrels or not doc_ids:
            return 0.0

        retrieved_docs = doc_ids[:k]

        dcg = 0.0
        for i, doc_id in enumerate(retrieved_docs):
            rel = self.qrels[query_id].get(doc_id, 0)
            # Using the common formula for DCG: rel_i / log_2(i+2)
            # i+2 because i is 0-indexed and we want log_2(rank+1) where rank starts at 1
            if rel == 2:
                dcg += rel / np.log2(i + 2)

        return dcg

    def ndcg_at_k(self,
                  query_id: str,
                  doc_ids: List[str],
                  k: int = 20) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain at k (NDCG@k) for a single query.

        Args:
            query_id: ID of the query
            doc_ids: List of retrieved document IDs (in ranked order)
            k: k value for NDCG@k

        Returns:
            NDCG@k value
        """
        if query_id not in self.qrels or not doc_ids:
            return 0.0

        # Calculate DCG@k
        dcg = self.dcg_at_k(query_id, doc_ids, k)

        # Calculate ideal DCG@k (IDCG@k)
        # Sort documents by relevance score (descending)
        relevant_docs = [(doc_id, rel) for doc_id, rel in self.qrels[query_id].items()]
        relevant_docs.sort(key=lambda x: x[1], reverse=True)

        ideal_doc_ids = [doc_id for doc_id, _ in relevant_docs]
        idcg = self.dcg_at_k(query_id, ideal_doc_ids, k)

        if idcg == 0:
            return 0.0

        return dcg / idcg

    def evaluate_run(self,
                     run_results: Dict[str, List[str]],
                     metrics: List[str] = ["p@20", "r@1000", "ndcg@20"]) -> Dict[str, Dict[str, float]]:
        """
        Evaluate a run (multiple queries) using various metrics.

        Args:
            run_results: Dictionary mapping query IDs to lists of retrieved document IDs
                         Format: {query_id: [doc_id1, doc_id2, ...], ...}
            metrics: List of metrics to compute

        Returns:
            Dictionary of evaluation results
            Format: {metric: {query_id: value, ..., "mean": mean_value}, ...}
        """
        logger.info(f"Evaluating run with {len(run_results)} queries")

        results = {}

        # Compute metrics for each query
        for metric in metrics:
            metric_name, k = self._parse_metric(metric)
            metric_results = {}

            for query_id, doc_ids in run_results.items():
                if metric_name == "p":
                    value = self.precision_at_k(query_id, doc_ids, k)
                elif metric_name == "r":
                    value = self.recall_at_k(query_id, doc_ids, k)
                elif metric_name == "ndcg":
                    value = self.ndcg_at_k(query_id, doc_ids, k)
                else:
                    logger.warning(f"Unknown metric: {metric}")
                    continue

                metric_results[query_id] = value

            # Compute mean metric across all queries
            mean_value = np.mean(list(metric_results.values())) if metric_results else 0.0
            metric_results["mean"] = mean_value

            results[metric] = metric_results
            logger.info(f"Mean {metric}: {mean_value:.4f}")

        return results

    def _parse_metric(self, metric: str) -> Tuple[str, int]:
        """
        Parse a metric string (e.g., "p@20") into a metric name and k value.

        Args:
            metric: Metric string

        Returns:
            Tuple of (metric_name, k)
        """
        metric_name, k = metric.split("@")
        return metric_name, int(k)

    def save_results(self,
                     results: Dict[str, Dict[str, float]],
                     run_name: str,
                     experiment: str = "default") -> None:
        """
        Save evaluation results to a JSON file.

        Args:
            results: Dictionary of evaluation results
            run_name: Name of the run
            experiment: Name of the experiment
        """
        experiment_dir = os.path.join(self.results_dir, experiment)
        os.makedirs(experiment_dir, exist_ok=True)

        file_path = os.path.join(experiment_dir, f"{run_name}.json")

        with open(file_path, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Saved evaluation results to {file_path}")

    def compare_runs(self,
                     experiment: str,
                     run_names: List[str],
                     metrics: List[str] = ["p@20", "r@1000", "ndcg@20"]) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Compare multiple runs.

        Args:
            experiment: Name of the experiment
            run_names: List of run names to compare
            metrics: List of metrics to compare

        Returns:
            Dictionary of comparison results
        """
        comparison = {}
        experiment_dir = os.path.join(self.results_dir, experiment)

        # Load results for each run
        for run_name in run_names:
            file_path = os.path.join(experiment_dir, f"{run_name}.json")

            if not os.path.exists(file_path):
                logger.warning(f"Results file not found: {file_path}")
                continue

            with open(file_path, "r") as f:
                results = json.load(f)

            comparison[run_name] = {metric: results.get(metric, {}).get("mean", 0.0)
                                    for metric in metrics}

        # Create comparison chart
        self.plot_comparison(comparison, metrics, experiment)

        return comparison

    def plot_comparison(self,
                        comparison: Dict[str, Dict[str, float]],
                        metrics: List[str],
                        experiment: str) -> None:
        """
        Plot a comparison of multiple runs.

        Args:
            comparison: Dictionary of comparison results
            metrics: List of metrics to compare
            experiment: Name of the experiment
        """
        runs = list(comparison.keys())

        fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 6))
        if len(metrics) == 1:
            axes = [axes]

        for i, metric in enumerate(metrics):
            ax = axes[i]
            values = [comparison[run].get(metric, 0.0) for run in runs]

            bar_positions = np.arange(len(runs))
            ax.bar(bar_positions, values, width=0.5)

            ax.set_title(f"{metric}")
            ax.set_xticks(bar_positions)
            ax.set_xticklabels(runs, rotation=45, ha="right")
            ax.set_ylabel("Score")
            ax.grid(axis="y", linestyle="--", alpha=0.7)

            # Add value labels on top of bars
            for j, v in enumerate(values):
                ax.text(j, v + 0.01, f"{v:.3f}", ha="center")

        plt.tight_layout()

        # Save the figure
        fig_path = os.path.join(self.results_dir, experiment, "comparison.png")
        plt.savefig(fig_path)
        plt.close()

        logger.info(f"Saved comparison chart to {fig_path}")


def load_qrels(qrels_data) -> Dict[str, Dict[str, int]]:
    """
    Load and format query relevance judgments (qrels) from a dataset.

    Args:
        qrels_data: Qrels dataset

    Returns:
        Dictionary mapping query IDs to dictionaries of document IDs and relevance scores
    """
    qrels = defaultdict(dict)

    for item in qrels_data:
        query_id = item["query-id"]
        doc_id = item["corpus-id"]
        relevance = item["score"]

        if relevance == 2:
            qrels[query_id][doc_id] = int(relevance)

    return dict(qrels)


def build_qrels_dicts(qrels_dataset):
    """
    Convert qrels_dataset into dictionaries for different relevance levels:
      - relevant_docs_by_query: judgment score == 1
      - highly_relevant_docs_by_query: judgment score == 2
      - overall_relevant_docs_by_query: judgment score > 0

    Args:
        qrels_dataset: Qrels dataset

    Returns:
        Tuple of (relevant_docs_by_query, highly_relevant_docs_by_query, overall_relevant_docs_by_query)
    """
    relevant_docs_by_query = defaultdict(set)
    highly_relevant_docs_by_query = defaultdict(set)
    overall_relevant_docs_by_query = defaultdict(set)

    for item in qrels_dataset:
        qid = item["query-id"]
        cid = item["corpus-id"]
        score = item["score"]

        if score == 1:
            relevant_docs_by_query[qid].add(cid)
        if score == 2:
            highly_relevant_docs_by_query[qid].add(cid)
        if score > 0:
            overall_relevant_docs_by_query[qid].add(cid)

    return relevant_docs_by_query, highly_relevant_docs_by_query, overall_relevant_docs_by_query


def main():
    """
    Test and demonstrate the Evaluator class functionality.
    """
    # Create a sample qrels dictionary
    sample_qrels = {
        "q1": {"doc1": 3, "doc2": 2, "doc3": 1, "doc4": 0},
        "q2": {"doc5": 2, "doc6": 1, "doc7": 1, "doc8": 0}
    }

    # Create a sample run results dictionary
    sample_run = {
        "q1": ["doc1", "doc3", "doc5", "doc2", "doc7"],
        "q2": ["doc5", "doc8", "doc7", "doc6", "doc1"]
    }

    # Initialize the evaluator
    evaluator = Evaluator(sample_qrels)

    # Evaluate the run
    results = evaluator.evaluate_run(sample_run)

    # Save the results
    evaluator.save_results(results, "sample_run", "test")

    # Compare with another run
    sample_run2 = {
        "q1": ["doc2", "doc1", "doc3", "doc5", "doc7"],
        "q2": ["doc6", "doc5", "doc7", "doc8", "doc1"]
    }

    results2 = evaluator.evaluate_run(sample_run2)
    evaluator.save_results(results2, "sample_run2", "test")

    # Compare the runs
    comparison = evaluator.compare_runs("test", ["sample_run", "sample_run2"])

    print("Comparison results:")
    for run, metrics in comparison.items():
        print(f"Run: {run}")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")


if __name__ == "__main__":
    main()