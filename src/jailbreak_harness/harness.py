import os
import json
import csv
import logging
import time
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

# Local imports from harness components
from .datatypes import TestCase, Variant # <-- Ensure Variant is imported
from .loader import load_test_suite_from_file

# Import necessary components from callers module
from .callers import BaseModelCaller, create_error_result 

# --- Setup ---
logger = logging.getLogger(__name__)

# --- Core Harness Class ---

class JailbreakHarness:
    """
    Main class for running test cases against a language model caller.
    """

    def __init__(self, model_caller: BaseModelCaller, model_name: str):
        """
        Initializes the harness with a model caller and the name of the model being tested.
        """
        if not isinstance(model_caller, BaseModelCaller):
            raise TypeError("model_caller must be an instance of BaseModelCaller or its subclass.")
        
        if not model_name or not isinstance(model_name, str):
             raise ValueError("model_name must be a non-empty string.")

        self.model_caller = model_caller
        self.model_name = model_name
        self.results: List[Dict[str, Any]] = []

    def run_variant(
        self, test: TestCase, variant: Variant, run_id: int
    ) -> Dict[str, Any]:
        """Runs a single variant and records the result."""
        ts = datetime.now(timezone.utc).isoformat()
        
        # Use variant temperature, falling back to 0.0 (deterministic) if None
        temp_value = variant.temperature if variant.temperature is not None else 0.0

        try:
            # Pass model_name as the first POSITIONAL argument
            resp = self.model_caller.call(
                self.model_name, 
                prompt=variant.prompt,
                system_note=variant.system_note,
                temperature=temp_value,
                max_tokens=2048, # Assuming standard max tokens
            )
        except Exception as e:
            logger.exception(f"Model caller failed internally")
            # Use the imported utility function 'create_error_result' directly
            resp = create_error_result(
                f"CallerInternalError_{type(e).__name__}", str(e)
            )

        # Standardized result dictionary structure
        result = {
            "timestamp": ts,
            "run_id": run_id,
            "test_id": test.id,
            "variant_id": variant.id,
            "model_name": self.model_name,
            "prompt": variant.prompt,
            "system_note": variant.system_note,
            "temperature": temp_value,
            "response": resp.get("text", "NO_RESPONSE"),
            "tokens": resp.get("tokens", 0),
            "safety_flags": resp.get("safety_flags", []),
            "meta": resp.get("meta", {}),
        }

        self.results.append(result)
        return result

    def run_tests(
        self,
        test_suite: List[TestCase],
        seeds: int = 1,
        temperatures: Optional[List[float]] = None,
        sleep: float = 0.0,
    ) -> None:
        """
        Runs all test cases in the suite across multiple seeds and temperature settings.
        """
        if not test_suite:
            raise ValueError("No tests provided in the test suite.")
        if seeds < 1:
            raise ValueError("seeds must be >= 1.")
        if sleep < 0.0:
            raise ValueError("sleep must be >= 0.")

        temp_list = temperatures if temperatures is not None else [0.0]
        
        # Validation for temperature list
        for t in temp_list:
            if not isinstance(t, (int, float)):
                 raise ValueError(f"All temperatures must be numeric. Found: {t}")

        total_runs = seeds * sum(
            len(t.variants)
            * (len(temp_list) if any(v.temperature is None for v in t.variants) else 1)
            for t in test_suite
        )

        run_count = 0
        logger.info(f"Starting test run for model '{self.model_name}'. Total runs: {total_runs}")

        for run_id in range(1, seeds + 1):
            logger.info(f"Seed {run_id}/{seeds}: Starting {len(test_suite)} test cases.")

            for t in test_suite:
                for variant in t.variants:
                    
                    # Determine temperatures to run for this variant
                    if variant.temperature is not None:
                        # Variant has a fixed temperature, run it once.
                        temp_values = [variant.temperature]
                    else:
                        # Variant has sweepable temperature (None), use the sweep list.
                        temp_values = temp_list

                    for temp in temp_values:
                        run_count += 1
                        
                        # FIX: Create a new Variant instance instead of using non-existent _replace()
                        # NOTE: This relies on the corrected dataclass order in datatypes.py
                        variant_for_call = Variant(
                            prompt=variant.prompt, # Mandatory field comes first
                            id=variant.id,
                            system_note=variant.system_note,
                            temperature=temp, # Use the current sweep temperature
                        )
                        
                        # Run the variant
                        self.run_variant(t, variant_for_call, run_id)
                        
                        # Sleep between runs if configured
                        if sleep > 0.0:
                            time.sleep(sleep)
            
            logger.info(f"Seed {run_id}/{seeds} completed.")


    def export_all(self, prefix: str):
        """
        Exports all recorded results to a timestamped CSV and JSON file.
        """
        if not self.results:
            logger.warning("No results recorded to export.")
            return

        ts_export = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. Export to JSON
        json_filename = f"{prefix}_{ts_export}.json"
        with open(json_filename, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=4)
        logger.info(f"Results exported to JSON: {json_filename}")

        # 2. Export to CSV
        csv_filename = f"{prefix}_{ts_export}.csv"
        
        # Determine all possible field names from all results for a complete header
        fieldnames = set()
        for result in self.results:
            # Flatten nested metadata (simplified approach)
            result_copy = result.copy()
            for key, value in result_copy.pop('meta', {}).items():
                fieldnames.add(f"meta_{key}")
            
            # Add top-level keys
            fieldnames.update(result_copy.keys())

        # Define a desired order for the main fields
        ordered_fields = [
            "timestamp", "run_id", "test_id", "variant_id", "model_name", 
            "temperature", "prompt", "system_note", "response", 
            "tokens", "safety_flags"
        ]
        
        # Create final field list: ordered fields + sorted remaining fields
        final_fieldnames = ordered_fields + sorted(list(fieldnames - set(ordered_fields)))

        with open(csv_filename, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=final_fieldnames)
            writer.writeheader()
            
            for result in self.results:
                # Prepare a flat row for CSV
                flat_row = result.copy()
                meta = flat_row.pop('meta', {})
                for key, value in meta.items():
                    flat_row[f"meta_{key}"] = value
                
                # Convert list fields to strings for CSV compatibility
                if isinstance(flat_row.get('safety_flags'), list):
                    flat_row['safety_flags'] = "|".join(flat_row['safety_flags'])
                
                writer.writerow(flat_row)
        
        logger.info(f"Results exported to CSV: {csv_filename}")

# --- Convenience Functions (Optional, but often helpful for testing/entry point) ---

def run_from_file(
    file_path: str,
    model_caller: BaseModelCaller,
    model_name: str,
    seeds: int = 1,
    temperatures: Optional[List[float]] = None,
    sleep: float = 0.0,
    output_prefix: str = "test_results",
) -> JailbreakHarness:
    """
    Loads a test suite from a file, runs the tests, and exports the results.
    """
    test_suite = load_test_suite_from_file(file_path)
    harness = JailbreakHarness(model_caller, model_name)
    harness.run_tests(test_suite, seeds, temperatures, sleep)
    harness.export_all(output_prefix)
    return harness
