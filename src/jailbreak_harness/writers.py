# src/jailbreak_harness/writers.py (Updated)

import csv
import json
import logging
import os
from typing import List, Dict, Any

# Import the utility function for consistent timestamped naming
from .reporter import get_timestamped_filename # <--- NEW IMPORT

logger = logging.getLogger(__name__)

# Define the standard fieldnames for CSV/JSON export
FIELDNAMES = [
    "run_id",
    "test_id",
    "test_name",
    "description",
    "variant_id",
    "prompt",
    "system_note",
    "temperature",
    "response",
    "tokens",
    "safety_flags",
    "retrieval_hits",
    "meta",
    "timestamp",
]


def save_json(results: List[Dict[str, Any]], filename_prefix: str, timestamp: str):
    """
    Saves results to a timestamped JSON file.
    
    Updated to use a pre-calculated timestamp for filename consistency.
    """
    # Use the shared utility to get the consistent filename
    filename, _ = get_timestamped_filename(filename_prefix, "json", timestamp) # <--- USE TIMESTAMP
    
    output_path = os.path.join(os.getcwd(), filename)
    
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4)
        logger.info(f"Results saved to JSON: {output_path}")
    except Exception as e:
        logger.error(f"Error writing JSON file {output_path}: {e}")


def save_csv(results: List[Dict[str, Any]], filename_prefix: str, timestamp: str):
    """
    Saves results to a timestamped CSV file.

    Updated to use a pre-calculated timestamp for filename consistency.
    """
    # Use the shared utility to get the consistent filename
    filename, _ = get_timestamped_filename(filename_prefix, "csv", timestamp) # <--- USE TIMESTAMP

    output_path = os.path.join(os.getcwd(), filename)
    
    try:
        with open(output_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()
            writer.writerows(results)
        logger.info(f"Results saved to CSV: {output_path}")
    except Exception as e:
        logger.error(f"Error writing CSV file {output_path}: {e}")


def export_all(results: List[Dict[str, Any]], filename_prefix: str, timestamp: str):
    """
    Exports results to both CSV and JSON formats using the same timestamp.
    
    Updated to receive a pre-calculated timestamp from the harness.
    """
    # Note: We don't need to generate the timestamp here, we just pass it down.
    save_json(results, filename_prefix, timestamp)
    save_csv(results, filename_prefix, timestamp)
