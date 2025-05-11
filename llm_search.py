import os
import json
import logging
import time
import re
from typing import Dict, List, Tuple, Any, Optional, Union
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from openai import OpenAI
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
import sys

# For Tavily API
from tavily import TavilyClient

# Import from our existing modules as needed
from search import SearchEngine
from evaluation import Evaluator
from query_expansion import QueryExpander
from llm_reranker import LLMReranker

from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("llm_search")

# Initialize Rich console for better CLI output
console = Console()


class TavilySearchEngine:
    """
    Class for performing web searches using the Tavily API client.
    """

    def __init__(self,
                 cache_dir: str = "cache",
                 tavily_results_dir: str = "cache/tavily_results"):
        """
        Initialize the Tavily search engine.

        Args:
            cache_dir: Base directory for caching
            tavily_results_dir: Directory for caching Tavily search results
        """
        self.cache_dir = cache_dir
        self.tavily_results_dir = tavily_results_dir

        # Create cache directories
        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(tavily_results_dir, exist_ok=True)

        # Initialize Tavily API key
        self.api_key = os.getenv("TAVILY_API_KEY")
        if not self.api_key:
            logger.warning("TAVILY_API_KEY not found in environment variables!")
        else:
            # Initialize Tavily client
            self.client = TavilyClient(self.api_key)

    def search(self,
               query: str,
               search_depth: str = "basic",
               max_results: int = 10,
               force_search: bool = False) -> Dict[str, Any]:
        """
        Perform a web search using the Tavily API.

        Args:
            query: Search query
            search_depth: Depth of search ("basic" or "advanced")
            max_results: Maximum number of results to return
            force_search: Whether to force a new search (ignore cache)

        Returns:
            Dictionary containing search results
        """
        # Prepare for caching
        query_hash = re.sub(r'[^a-zA-Z0-9]', '_', query.lower())
        cache_file = os.path.join(self.tavily_results_dir, f"search_{query_hash}.json")

        # Check cache only if not forcing search
        if os.path.exists(cache_file) and not force_search:
            logger.info(f"Loading Tavily search results from cache for '{query}'")
            with open(cache_file, 'r') as f:
                return json.load(f)

        # Check if API key is available
        if not self.api_key:
            logger.error("No Tavily API key found. Please set the TAVILY_API_KEY environment variable.")
            return {"error": "No API key", "results": []}

        logger.info(f"Performing Tavily web search for query: '{query}'")

        try:
            # Use the Tavily client to perform the search
            response = self.client.search(
                query=query,
                search_depth=search_depth,
                max_results=max_results,
                include_answer=True
            )

            # Cache results
            with open(cache_file, 'w') as f:
                json.dump(response, f, indent=2)

            return response

        except Exception as e:
            logger.error(f"Error performing Tavily search: {e}")
            return {"error": str(e), "results": []}


class LLMSearchEngine:
    """
    Class for performing web search + LLM-based QA with explanations.
    """

    def __init__(self,
                 cache_dir: str = "cache",
                 qa_results_dir: str = "cache/qa_results",
                 model: str = "gpt-4o-mini"):
        """
        Initialize the LLM search engine.

        Args:
            cache_dir: Base directory for caching
            qa_results_dir: Directory for caching QA results
            model: OpenAI model to use for QA
        """
        self.cache_dir = cache_dir
        self.qa_results_dir = qa_results_dir
        self.model = model

        # Create cache directories
        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(qa_results_dir, exist_ok=True)

        # Initialize Tavily search engine
        self.tavily = TavilySearchEngine()

        # Initialize OpenAI client
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.warning("OPENAI_API_KEY not found in environment variables!")
        self.client = OpenAI(api_key=api_key)

        # Define the QA prompt with recommended follow-up queries
        self.qa_prompt = """
        Here are some web search results for a query.

        1. Based on this information, answer the question.
        2. Provide a 3-line rationale based on the information.
        3. Suggest 3-4 related follow-up queries the user might be interested in exploring next.

        Query: {query}

        Web Results:
        {web_results}

        Return as JSON: {{"answer": "...", "explanation": "...", "recommended_queries": ["query1", "query2", "query3"]}}
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
                temperature=0.3,  # Lower temperature for more consistent answers
                max_tokens=1024,
                response_format={"type": "json_object"}  # Ensure JSON format
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {e}")
            return ""

    def search_and_answer(self,
                          query: str,
                          search_depth: str = "basic",
                          max_results: int = 7,
                          force_search: bool = False,
                          force_answer: bool = False) -> Dict[str, Any]:
        """
        Search the web and generate an answer with explanation and recommended follow-up queries.

        Args:
            query: User query
            search_depth: Depth of search ("basic" or "advanced")
            max_results: Maximum number of search results to use
            force_search: Whether to force a new search (ignore cache)
            force_answer: Whether to force a new answer (ignore cache)

        Returns:
            Dictionary containing the answer, explanation, recommended_queries, and search results
        """
        # Prepare for caching
        query_hash = re.sub(r'[^a-zA-Z0-9]', '_', query.lower())
        cache_file = os.path.join(self.qa_results_dir, f"qa_{query_hash}.json")

        # Check cache only if not forcing answer
        if os.path.exists(cache_file) and not force_answer:
            logger.info(f"Loading QA results from cache for '{query}'")
            with open(cache_file, 'r') as f:
                return json.load(f)

        # Step 1: Search the web - passing force_search flag
        search_results = self.tavily.search(query, search_depth, max_results, force_search)

        # Check for search errors
        if "error" in search_results and search_results["error"]:
            logger.error(f"Search error: {search_results['error']}")
            return {
                "query": query,
                "answer": "Sorry, I encountered an error while searching the web.",
                "explanation": f"Error: {search_results.get('error', 'Unknown error')}",
                "recommended_queries": ["Try a more specific query",
                                        "Try rephrasing your question",
                                        "Check your internet connection"],
                "search_results": []
            }

        # Format search results for the prompt
        web_results_text = ""
        search_results_list = search_results.get("results", [])

        for i, result in enumerate(search_results_list):
            title = result.get("title", "No title")
            content = result.get("content", "No content")
            url = result.get("url", "No URL")

            # Format each result
            web_results_text += f"- Title: {title}\n  Snippet: {content}\n  URL: {url}\n\n"

        # Step 2: Generate answer with LLM
        prompt = self.qa_prompt.format(
            query=query,
            web_results=web_results_text
        )

        # Call OpenAI API
        llm_response = self._call_openai(prompt)

        try:
            # Parse JSON response
            qa_result = json.loads(llm_response)

            # Create complete result with recommended_queries instead of recommendation
            complete_result = {
                "query": query,
                "answer": qa_result.get("answer", "No answer provided"),
                "explanation": qa_result.get("explanation", "No explanation provided"),
                "recommended_queries": qa_result.get("recommended_queries", []),
                "search_results": search_results_list,
                "tavily_answer": search_results.get("answer", "")
            }

            # Cache result
            with open(cache_file, 'w') as f:
                json.dump(complete_result, f, indent=2)

            return complete_result

        except Exception as e:
            logger.error(f"Error parsing LLM response: {e}")
            logger.error(f"Response: {llm_response}")

            # Return error result
            error_result = {
                "query": query,
                "answer": "Sorry, I couldn't generate a proper answer.",
                "explanation": f"Error parsing LLM response: {str(e)}",
                "recommended_queries": ["Try a simpler query",
                                        "Try asking a more specific question",
                                        "Try again later"],
                "search_results": search_results_list,
                "tavily_answer": search_results.get("answer", "")
            }

            # Still cache the error result to avoid repeated failures
            with open(cache_file, 'w') as f:
                json.dump(error_result, f, indent=2)

            return error_result

    def save_results_for_evaluation(self,
                                    results: Dict[str, Any],
                                    output_dir: str = "results/web_qa"):
        """
        Save results in a format suitable for manual evaluation.

        Args:
            results: Dictionary containing QA results
            output_dir: Output directory for evaluation results
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Prepare for file naming
        query = results["query"]
        query_hash = re.sub(r'[^a-zA-Z0-9]', '_', query.lower())
        output_file = os.path.join(output_dir, f"qa_{query_hash}.md")

        # Format results as markdown for better readability
        with open(output_file, 'w') as f:
            f.write(f"# Query: {query}\n\n")

            f.write("## Answer\n\n")
            f.write(f"{results.get('answer', 'No answer provided')}\n\n")

            f.write("## Explanation\n\n")
            f.write(f"{results.get('explanation', 'No explanation provided')}\n\n")

            f.write("## Recommended Follow-up Queries\n\n")
            recommended_queries = results.get("recommended_queries", [])
            if recommended_queries:
                for i, rec_query in enumerate(recommended_queries):
                    f.write(f"{i + 1}. {rec_query}\n")
            else:
                f.write("No recommended queries provided.\n")
            f.write("\n")

            f.write("## Search Results\n\n")
            for i, result in enumerate(results.get("search_results", [])):
                title = result.get("title", "No title")
                snippet = result.get("content", "No content")
                url = result.get("url", "No URL")

                f.write(f"### {i + 1}. {title}\n\n")
                f.write(f"**URL**: {url}\n\n")
                f.write(f"{snippet}\n\n")
                f.write("---\n\n")

            # If Tavily provided its own answer
            tavily_answer = results.get("tavily_answer")
            if tavily_answer:
                f.write("## Tavily Generated Answer\n\n")
                f.write(f"{tavily_answer}\n\n")

        logger.info(f"Saved evaluation results to {output_file}")

        # Also save a JSON version for programmatic analysis
        json_file = os.path.join(output_dir, f"qa_{query_hash}.json")
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=2)

        return output_file


def display_results_in_cli(results: Dict[str, Any]):
    """
    Display search and QA results in the CLI with rich formatting.

    Args:
        results: Dictionary containing QA results
    """
    console.print(f"\n[bold cyan]Query:[/bold cyan] {results['query']}\n")

    # Display answer
    console.print(Panel(
        results.get('answer', 'No answer provided'),
        title="[bold green]Answer[/bold green]",
        expand=False
    ))

    # Display explanation
    console.print(Panel(
        results.get('explanation', 'No explanation provided'),
        title="[bold yellow]Explanation[/bold yellow]",
        expand=False
    ))

    # Display recommended queries
    recommended_queries = results.get("recommended_queries", [])
    if recommended_queries:
        rec_text = "You might also want to explore:\n\n"
        for i, rec_query in enumerate(recommended_queries):
            rec_text += f"{i + 1}. {rec_query}\n"

        console.print(Panel(
            rec_text,
            title="[bold magenta]Recommended Follow-up Queries[/bold magenta]",
            expand=False
        ))

    # Display search results
    console.print("\n[bold blue]Search Results:[/bold blue]\n")

    for i, result in enumerate(results.get("search_results", [])[:5]):  # Show just top 5 for readability
        title = result.get("title", "No title")
        url = result.get("url", "No URL")

        console.print(f"[bold]{i + 1}. {title}[/bold]")
        console.print(f"[italic cyan]{url}[/italic cyan]")
        console.print(result.get("content", "No content")[:300] + "...\n")


def interactive_mode(search_engine, suggested_queries, force_search, force_answer):
    """
    Run the LLM search engine in interactive mode, allowing users to input queries.

    Args:
        search_engine: LLMSearchEngine instance
        suggested_queries: List of suggested queries
        force_search: Whether to force new searches (ignore cache)
        force_answer: Whether to force new answers (ignore cache)
    """
    console.print("[bold green]LLM + Internet Search QA System[/bold green]")
    console.print("Type 'exit' or 'quit' to exit the program.\n")

    if force_search or force_answer:
        console.print(
            f"[bold yellow]Running with force_search={force_search}, force_answer={force_answer}[/bold yellow]\n")

    # Display suggested queries
    console.print("[bold yellow]Suggested Queries:[/bold yellow]")
    for i, query in enumerate(suggested_queries):
        console.print(f"{i + 1}. {query}")
    console.print()

    while True:
        try:
            # Get user input
            user_query = console.input(
                "[bold cyan]Enter your query (or a number 1-5 for suggested queries): [/bold cyan]")

            # Check for exit command
            if user_query.lower() in ('exit', 'quit'):
                console.print("[bold green]Exiting program. Goodbye![/bold green]")
                break

            # Check if user entered a number for suggested queries
            if user_query.isdigit() and 1 <= int(user_query) <= len(suggested_queries):
                query_idx = int(user_query) - 1
                user_query = suggested_queries[query_idx]
                console.print(f"Selected: [bold]{user_query}[/bold]\n")

            # Search and answer - passing the force flags
            with console.status("[bold green]Searching and generating answer...[/bold green]"):
                results = search_engine.search_and_answer(
                    user_query,
                    force_search=force_search,
                    force_answer=force_answer
                )

            # Display results
            display_results_in_cli(results)

            # Save results for evaluation
            output_file = search_engine.save_results_for_evaluation(results)
            console.print(f"\n[italic]Results saved to: {output_file}[/italic]\n")

        except KeyboardInterrupt:
            console.print("\n[bold red]Operation cancelled by user.[/bold red]")
            break

        except Exception as e:
            console.print(f"\n[bold red]Error: {str(e)}[/bold red]")
            logger.error(f"Error in interactive mode: {e}")
            continue

        console.print("\n" + "-" * 80 + "\n")


def batch_mode(search_engine, queries, output_dir, force_search, force_answer):
    """
    Run the LLM search engine on a batch of queries.

    Args:
        search_engine: LLMSearchEngine instance
        queries: List of queries to process
        output_dir: Output directory for evaluation results
        force_search: Whether to force new searches (ignore cache)
        force_answer: Whether to force new answers (ignore cache)
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    if force_search or force_answer:
        logger.info(f"Running batch mode with force_search={force_search}, force_answer={force_answer}")

    # Process each query
    results = []
    for query in tqdm(queries, desc="Processing queries"):
        try:
            # Search and answer - passing the force flags
            qa_result = search_engine.search_and_answer(
                query,
                force_search=force_search,
                force_answer=force_answer
            )

            # Save results for evaluation
            output_file = search_engine.save_results_for_evaluation(qa_result, output_dir)

            # Store results for summary
            results.append({
                "query": query,
                "output_file": output_file,
                "success": True
            })

        except Exception as e:
            logger.error(f"Error processing query '{query}': {e}")
            results.append({
                "query": query,
                "error": str(e),
                "success": False
            })

    # Create a summary report
    summary_file = os.path.join(output_dir, "batch_summary.md")
    with open(summary_file, 'w') as f:
        f.write("# Batch Processing Summary\n\n")
        f.write(f"Processed {len(queries)} queries\n\n")

        if force_search or force_answer:
            f.write(f"**Force flags**: force_search={force_search}, force_answer={force_answer}\n\n")

        f.write("## Successful Queries\n\n")
        successful = [r for r in results if r["success"]]
        for result in successful:
            f.write(f"- {result['query']} ([results]({os.path.basename(result['output_file'])}))\n")

        f.write("\n## Failed Queries\n\n")
        failed = [r for r in results if not r["success"]]
        for result in failed:
            f.write(f"- {result['query']}: {result.get('error', 'Unknown error')}\n")

    logger.info(f"Batch processing complete. Summary saved to {summary_file}")
    return summary_file


def visualize_results(output_dir="results/web_qa"):
    """
    Create visualizations for evaluation of search results.

    Args:
        output_dir: Directory containing QA results
    """
    # Check if directory exists
    if not os.path.exists(output_dir):
        logger.error(f"Output directory {output_dir} does not exist")
        return

    # Find all JSON result files
    json_files = [f for f in os.listdir(output_dir) if f.endswith('.json')]

    if not json_files:
        logger.error(f"No JSON files found in {output_dir}")
        return

    # Load results
    all_results = []
    for file in json_files:
        with open(os.path.join(output_dir, file), 'r') as f:
            try:
                result = json.load(f)
                all_results.append(result)
            except Exception as e:
                logger.error(f"Error loading {file}: {e}")

    # Create visualizations directory
    viz_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)

    # 1. Bar chart of search result counts per query
    plt.figure(figsize=(12, 6))
    queries = [result["query"][:30] + "..." if len(result["query"]) > 30 else result["query"]
               for result in all_results]
    result_counts = [len(result.get("search_results", [])) for result in all_results]

    plt.bar(range(len(queries)), result_counts)
    plt.xticks(range(len(queries)), queries, rotation=45, ha="right")
    plt.xlabel("Query")
    plt.ylabel("Number of Search Results")
    plt.title("Search Results per Query")
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, "search_result_counts.png"))
    plt.close()

    # 2. Length analysis of answers, explanations, and recommended queries count
    plt.figure(figsize=(12, 6))

    answer_lengths = [len(result.get("answer", "")) for result in all_results]
    explanation_lengths = [len(result.get("explanation", "")) for result in all_results]
    rec_query_counts = [len(result.get("recommended_queries", [])) for result in all_results]

    x = np.arange(len(queries))
    width = 0.25

    plt.bar(x - width, answer_lengths, width, label="Answer Length")
    plt.bar(x, explanation_lengths, width, label="Explanation Length")
    plt.bar(x + width, rec_query_counts, width, label="# Recommended Queries")

    plt.xticks(x, queries, rotation=45, ha="right")
    plt.xlabel("Query")
    plt.ylabel("Character Length / Count")
    plt.title("Response Component Analysis")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, "response_analysis.png"))
    plt.close()

    # Create a summary HTML file
    html_file = os.path.join(viz_dir, "summary.html")
    with open(html_file, 'w') as f:
        f.write("<html><head><title>Web QA Results Summary</title>")
        f.write("<style>body{font-family:Arial;margin:20px}table{border-collapse:collapse;width:100%}")
        f.write("th,td{padding:8px;text-align:left;border-bottom:1px solid #ddd}</style></head>")
        f.write("<body><h1>Web QA Results Summary</h1>")

        # Add visualizations
        f.write("<h2>Visualizations</h2>")
        f.write("<img src='search_result_counts.png' style='max-width:100%'><br><br>")
        f.write("<img src='response_analysis.png' style='max-width:100%'><br><br>")

        # Add query summary table
        f.write("<h2>Query Summary</h2>")
        f.write("<table><tr><th>Query</th><th>Results</th><th>Answer Length</th><th>Recommended Queries</th></tr>")

        for result in all_results:
            query = result["query"]
            num_results = len(result.get("search_results", []))
            answer_len = len(result.get("answer", ""))
            rec_queries = ", ".join(result.get("recommended_queries", [])[:3])

            f.write(f"<tr><td>{query}</td><td>{num_results}</td><td>{answer_len}</td><td>{rec_queries}</td></tr>")

        f.write("</table></body></html>")

    logger.info(f"Visualizations created in {viz_dir}")
    logger.info(f"Summary HTML: {html_file}")
    return html_file


def process_single_query(search_engine, query, output_dir, force_search, force_answer):
    """
    Process a single query and display the results.

    Args:
        search_engine: LLMSearchEngine instance
        query: Query to process
        output_dir: Output directory for evaluation results
        force_search: Whether to force new searches (ignore cache)
        force_answer: Whether to force new answers (ignore cache)
    """
    console.print(f"[bold]Processing query:[/bold] {query}")

    if force_search or force_answer:
        console.print(f"[bold yellow]Using force_search={force_search}, force_answer={force_answer}[/bold yellow]")

    # Search and answer - passing the force flags
    with console.status("[bold green]Searching and generating answer...[/bold green]"):
        results = search_engine.search_and_answer(
            query,
            force_search=force_search,
            force_answer=force_answer
        )

    # Display results
    display_results_in_cli(results)

    # Save results for evaluation
    output_file = search_engine.save_results_for_evaluation(results, output_dir)
    console.print(f"\n[italic]Results saved to: {output_file}[/italic]\n")

    return results


def main():
    """
    Main function to run the LLM search engine.
    """
    # Static configuration variables
    mode = "visualize"  # Options: "interactive", "batch", "single", "visualize"
    single_query = "Should I wear a mask in April 2020?"  # Query for single mode
    force_search = False  # Whether to force new searches (ignore cache)
    force_answer = False  # Whether to force new answers (ignore cache)
    output_dir = "results/web_qa"  # Directory for evaluation results
    openai_model = "gpt-4o-mini"  # OpenAI model to use
    search_depth = "basic"  # Tavily search depth ("basic" or "advanced")
    max_results = 7  # Maximum number of search results to use

    # Suggested queries from the project instructions
    suggested_queries = [
        "Should I wear a mask in April 2020?",
        "Can COVID spread through surfaces?",
        "What is the link between COVID and diabetes?",
        "Are vaccines effective against Delta variant?",
        "Can drinking turmeric prevent COVID-19?"
    ]

    # Log start time
    start_time = time.time()

    # Check for API keys
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY environment variable not set!")
        return

    if not os.getenv("TAVILY_API_KEY"):
        logger.error("TAVILY_API_KEY environment variable not set!")
        return

    # Initialize search engine
    search_engine = LLMSearchEngine(model=openai_model)

    # Run in the selected mode
    if mode == "interactive":
        interactive_mode(search_engine, suggested_queries, force_search, force_answer)

    elif mode == "batch":
        batch_mode(search_engine, suggested_queries, output_dir, force_search, force_answer)

    elif mode == "single":
        process_single_query(search_engine, single_query, output_dir, force_search, force_answer)

    elif mode == "visualize":
        visualize_results(output_dir)

    else:
        logger.error(f"Unknown mode: {mode}")

    # Log total execution time
    execution_time = time.time() - start_time
    logger.info(f"Total execution time: {execution_time:.2f} seconds")


if __name__ == "__main__":
    main()