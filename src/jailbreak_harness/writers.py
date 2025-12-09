# writers.py

import os
import csv
import json
import logging

logger = logging.getLogger(__name__)

def write_csv(results, filename_prefix, get_timestamped_filename):
    if not results:
        logger.warning("No results to save.")
        return

    filename_relative = get_timestamped_filename(filename_prefix, "csv")
    filename = os.path.join(os.getcwd(), filename_relative)

    csv_fields = [
        "run_id",
        "test_id",
        "test_name",
        "variant_id",
        "temperature",
        "response",
        "safety_flags",
        "tokens",
        "timestamp",
        "prompt",
        "system_note",
        "description",
        "retrieval_hits",
        "meta",
    ]

    extra_keys = sorted(
        set().union(*(r.keys() for r in results)) - set(csv_fields)
    )
    all_fields = csv_fields + extra_keys

    try:
        with open(filename, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=all_fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(results)
        logger.info(f"Saved {len(results)} results to CSV: {filename}")
    except IOError as e:
        logger.critical(f"Failed to write CSV file: {e}")
        raise


def write_json(results, filename_prefix, get_timestamped_filename):
    if not results:
        logger.warning("No results to save.")
        return

    filename_relative = get_timestamped_filename(filename_prefix, "json")
    filename = os.path.join(os.getcwd(), filename_relative)

    try:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved {len(results)} results to JSON: {filename}")
    except IOError as e:
        logger.critical(f"Failed to write JSON file: {e}")
        raise


def export_all(results, filename_prefix, get_timestamped_filename):
    write_csv(results, filename_prefix, get_timestamped_filename)
    write_json(results, filename_prefix, get_timestamped_filename)

