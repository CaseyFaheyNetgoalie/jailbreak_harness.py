# src/jailbreak_harness/harness.py

__version__ = "0.2.1"

import os
import time
import random
import logging
import traceback
import json
from datetime import datetime, timezone
from tqdm import tqdm
import argparse
from typing import Dict, Any, List, Optional, Tuple # Ensure these are present

from .writers import export_all
from .callers import MockModelCaller, OpenAIModelCaller, HFModelCaller, BaseModelCaller
from .datatypes import TestCase, Variant
from .loader import load_test_suite_from_file
from .reporter import generate_summary, get_timestamped_filename
from .example_suite import TEST_SUITE

# ---------------- Logging ----------------
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("jailbreak_harness.log", mode="w"),
            logging.StreamHandler(),
        ],
    )
                    
# ---------------- Harness ----------------
class JailbreakHarness:
    """Core harness to run test cases against a model caller."""

    def __init__(self, model_caller: BaseModelCaller, model_name: str = "mock-model"):
        if not isinstance(model_caller, BaseModelCaller):
            raise TypeError("model_caller must be an instance of BaseModelCaller")
        if not model_name or not isinstance(model_name, str):
            raise ValueError("model_name must be a non-empty string")

        self.model_caller = model_caller
        self.model_name = model_name
        self.results: List[Dict[str, Any]] = []
        self._last_export_ts: Optional[str] = None

    # REMOVED: def _get_timestamped_filename(self, prefix: str, ext: str) -> str:
    # REMOVED: Logic moved to reporter.py

    def run_variant(
        self, test: TestCase, variant: Variant, run_id: int
    ) -> Dict[str, Any]:
        """Runs a single variant and records the result."""
        ts = datetime.now(timezone.utc).isoformat()
        temp_value = variant.temperature if variant.temperature is not None else 0.0

        try:
            resp = self.model_caller.call(
                model=self.model_name,
                prompt=variant.prompt,
                system=variant.system_note,
                temperature=temp_value,
            )
        except Exception as e:
            logger.exception(f"Model caller failed internally")
            resp = self.model_caller.format_error_response(
                f"CallerInternalError_{type(e).__name__}", str(e)
            )

        record = {
            "run_id": run_id,
            "test_id": test.id,
            "test_name": test.name,
            "description": test.description,
            "variant_id": variant.id,
            "prompt": variant.prompt,
            "system_note": variant.system_note,
            "temperature": temp_value,
            "response": resp.get("text"),
            "tokens": resp.get("tokens"),
            "safety_flags": ",".join(resp.get("safety_flags", [])),
            "retrieval_hits": json.dumps(
                resp.get("meta", {}).get("retrieval_hits", [])
            ),
            "meta": json.dumps(resp.get("meta", {})),
            "timestamp": ts,
        }

        self.results.append(record)
        return record

    def run_tests(
        self,
        tests: List[TestCase],
        seeds: int = 1,
        temperatures: Optional[List[float]] = None,
        sleep: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """Runs a full test suite with seeds and temperature variations."""
        # Input validation
        if not tests:
            raise ValueError("No tests provided")
        if seeds < 1:
            raise ValueError("seeds must be >= 1")
        if sleep < 0:
            raise ValueError("sleep must be >= 0")

        temperatures_to_run = temperatures or [0.0]

        # Validate temperatures
        for temp in temperatures_to_run:
            if not isinstance(temp, (int, float)):
                raise ValueError(f"All temperatures must be numeric, got: {temp}")
            if not 0.0 <= temp <= 2.0:
                logger.warning(
                    f"Temperature {temp} outside typical range [0.0, 2.0]"
                )

        run_id = 0
        self._last_export_ts = None

        # Flatten all test variants with effective temperature
        variant_runs: List[Tuple[TestCase, Variant]] = []
        for t in tests:
            for v in t.variants:
                run_temps = (
                    [v.temperature]
                    if v.temperature is not None
                    else temperatures_to_run
                )

                for temp in run_temps:
                    variant_runs.append(
                        (
                            t,
                            Variant(
                                id=v.id,
                                prompt=v.prompt,
                                system_note=v.system_note,
                                temperature=temp,
                            ),
                        )
                    )

        total_runs = len(variant_runs) * seeds
        logger.info(f"Starting test run: {total_runs} total executions")

        for seed in range(seeds):
            random.seed(seed)
            logger.info(f"--- Seed {seed + 1}/{seeds} ---")

            for t, variant in tqdm(
                variant_runs, desc=f"Seed {seed + 1}/{seeds}", unit="test"
            ):
                run_id += 1
                self.run_variant(t, variant, run_id)
                if sleep > 0:
                    time.sleep(sleep)

        logger.info(f"Completed {run_id} test executions")
        return self.results

    def export_all(self, filename_prefix: str = "jailbreak_results"):
        """Ensures a single timestamp is used and calls the external exporter."""
        if not self.results:
            logger.warning("export_all called with empty results")
            return
        
        # Use the reporter utility to generate/cache the timestamp
        _, self._last_export_ts = get_timestamped_filename(
            filename_prefix, "dummy", self._last_export_ts
        )
        
        # Call the external export_all function with the cached timestamp
        export_all(
            results=self.results,
            filename_prefix=filename_prefix,
            timestamp=self._last_export_ts,
        )

    # REMOVED: def summary(self):
    # REMOVED: Logic moved to reporter.py
    
# ---------------- Main Runner ----------------
# ... (parse_args and main() remain the same, as they already call the external functions)
