"""Caching utilities.

TODO
"""

from typing import Any, Callable, List, Optional
import psutil
import multiprocessing as mp
import gc
from tqdm import tqdm


def batch_func(
    list_inputs: List[Any], function: Callable[[Any], Any]
) -> List[Any]:
    """Process a batch of inputs with inner progress bar."""
    results = []
    # Inner progress bar for items in this chunk
    for item in tqdm(
        list_inputs,
        desc="Items in chunk",
        leave=False,  # Don't leave this bar when done
        position=1,  # Show below the main bar
    ):
        results.append(function(item))
    return results


def chunked_parallel(
    input_list: List[Any],
    function: Callable[[Any], Any],
    chunks: int = 20,
    max_cpu: int = 16,
    output_func: Optional[Callable[[List[Any]], None]] = None,
    **kwargs: Any,
) -> Optional[List[Any]]:
    """Optimized version of chunked_parallel with both outer and inner progress
    tracking.

    Args:
        input_list: List of inputs to process
        function: Function to apply to each input
        chunks: Number of chunks to split input into
        max_cpu: Maximum number of CPU workers to use
        output_func: Optional function to call with results as they come in.
                     If provided, results are not accumulated and None is returned.
        **kwargs: Additional keyword arguments

    Returns:
        List of results if output_func is None, otherwise None
    """
    list_len = len(input_list)
    if list_len == 0:
        raise ValueError("Empty list to process!")

    # Optimize chunk size
    optimal_chunks = min(chunks, max(1, list_len // (max_cpu * 2)))
    chunk_size = max(1, list_len // optimal_chunks)

    chunked_list = [
        input_list[i : i + chunk_size] for i in range(0, list_len, chunk_size)
    ]

    # Configure process pool
    cpus = min(psutil.cpu_count(logical=False) or 1, max_cpu)

    print(
        f"\nProcessing with {cpus} workers, {len(chunked_list)} chunks of ~{chunk_size} items each"
    )

    with mp.Pool(processes=cpus) as pool:
        try:
            # Main progress bar for chunks
            # Use functools.partial to bind the function to batch_func
            from functools import partial

            processor = partial(batch_func, function=function)

            if output_func is not None:
                # Stream results to output function without accumulating
                for result in tqdm(
                    pool.imap(processor, chunked_list),
                    total=len(chunked_list),
                    desc="Processing chunks",
                    position=0,
                ):
                    output_func([result])
                return None
            else:
                # Collect all results
                results = list(
                    tqdm(
                        pool.imap(processor, chunked_list),
                        total=len(chunked_list),
                        desc="Processing chunks",
                        position=0,  # Keep at top
                    )
                )

                return [item for sublist in results for item in sublist]

        finally:
            pool.close()
            pool.join()
            gc.collect()
            print("\n" * 2)  # Clear progress bar lines
