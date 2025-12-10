# src/jailbreak_harness/reporter.py

import json
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

# --- Utility Functions ---

def get_timestamped_filename(prefix: str, ext: str, timestamp_cache: Optional[str]) -> tuple[str, str]:
    """
    Generates a UTC timestamped filename and returns the name and the timestamp used.

    Args:
        prefix: The file name prefix (e.g., 'jailbreak_results').
        ext: The file extension (e.g., 'csv', 'json').
        timestamp_cache: A cached timestamp string (YYYYMMDDTHHMMSS) or None.

    Returns:
        A tuple: (timestamped_filename, used_timestamp_string)
    """
    if timestamp_cache is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    else:
        timestamp = timestamp_cache
        
    return f"{prefix}_{timestamp}.{ext}", timestamp

# --- Reporting Functions ---

def generate_summary(results: List[Dict[str, Any]], model_name: str):
    """Logs a brief summary of test results to the console."""
    if not results:
        logger.info("No results to summarize")
        return

    by_test = {}
    total_runs = len(results)
    safety_triggered = 0
    errors = 0

    for r in results:
        tid = r["test_id"]
        by_test.setdefault(
            tid,
            {"runs": 0, "unique_responses": set(), "safety_flags": 0, "errors": 0},
        )
        by_test[tid]["runs"] += 1
        
        # Check if response exists before adding to set (handles caller internal errors that return minimal dict)
        response_text = r.get("response", "N/A")
        by_test[tid]["unique_responses"].add(response_text)

        if r.get("safety_flags"):
            by_test[tid]["safety_flags"] += 1
            safety_triggered += 1
            
        # Check for error indicator in the response text (based on model caller format_error_response)
        if response_text and "ERROR:" in response_text:
            by_test[tid]["errors"] += 1
            errors += 1

    logger.info("\n" + "=" * 60)
    logger.info("TEST RUN SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Model: {model_name}")
    logger.info(f"Total executions: {total_runs}")
    logger.info(f"Unique tests: {len(by_test)}")
    # Handle division by zero edge case if total_runs is 0 (though should be caught earlier)
    safety_percentage = (safety_triggered / total_runs * 100) if total_runs else 0.0
    error_percentage = (errors / total_runs * 100) if total_runs else 0.0
    
    logger.info(
        f"Safety flags triggered: {safety_triggered} ({safety_percentage:.1f}%)"
    )
    logger.info(f"Errors encountered: {errors} ({error_percentage:.1f}%)")

    logger.info("\n--- Per-Test Summary ---")
    for tid, info in sorted(by_test.items()):
        logger.info(
            f"{tid}: {info['runs']} runs, "
            f"{len(info['unique_responses'])} unique responses, "
            f"{info['safety_flags']} safety flags, "
            f"{info['errors']} errors"
        )
