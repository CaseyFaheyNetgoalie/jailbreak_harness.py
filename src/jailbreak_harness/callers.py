# jailbreak_harness/callers.py
import os
import random
import logging
from typing import Dict, Any, Optional

# Attempt to import external dependencies for specific callers
try:
    import openai
    from openai import OpenAI
    # Import specific error types for cleaner error handling
    from openai import APIConnectionError, RateLimitError, AuthenticationError, NotFoundError, APIStatusError
except ImportError:
    # This is fine; only the OpenAI caller will fail initialization if missing.
    pass

try:
    import requests
    # Import specific exception types for cleaner error handling
    from requests.exceptions import HTTPError, Timeout, RequestException
except ImportError:
    # This is fine; only the HF caller will fail initialization if missing.
    pass

logger = logging.getLogger(__name__)


class BaseModelCaller:
    """Interface for LLM model callers. Override call() in subclasses."""

    def call(
        self,
        model: str,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Call the model with given parameters.
        
        Args:
            model: Model identifier
            prompt: User prompt
            system: Optional system message
            temperature: Sampling temperature
            
        Returns:
            Dict with keys: 'text', 'tokens', 'safety_flags', 'meta'
        """
        raise NotImplementedError()

    def format_error_response(
        self, error_type: str, details: str = ""
    ) -> Dict[str, Any]:
        """Helper to create a standard error dict."""
        error_msg = f"ERROR: {error_type}"
        if details:
            error_msg += f" ({details})"
        return {
            "text": error_msg,
            "tokens": 0,
            "safety_flags": [error_type],
            "meta": {"error": True, "details": details},
        }

    def validate_inputs(self, model: str, prompt: str, temperature: float):
        """Validate common inputs."""
        if not model or not isinstance(model, str):
            raise ValueError("Model must be a non-empty string")
        if not prompt or not isinstance(prompt, str):
            raise ValueError("Prompt must be a non-empty string")
        if not isinstance(temperature, (int, float)):
            raise ValueError("Temperature must be numeric")
        if not 0.0 <= temperature <= 2.0:
            logger.warning(
                f"Temperature {temperature} outside typical range [0.0, 2.0]"
            )


class MockModelCaller(BaseModelCaller):
    """Deterministic, safe mock for demos or CI testing."""

    def __init__(self, seed: int = 42):
        random.seed(seed)
        self.call_count = 0
        logger.info("MockModelCaller initialized.") # Added initialization log

    def call(
        self,
        model: str,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        """Mock implementation with pattern matching."""
        self.validate_inputs(model, prompt, temperature)
        self.call_count += 1

        markers = [
            "LAB_OK",
            "NF_OK",
            "DOC_OK",
            "FILE_OK",
            "ENC_OK",
            "MEM_OK",
            "COT_OK",
            "CANARY_TOKEN",
        ]
        text = "REFUSE"
        safety_flags = []
        meta = {"call_count": self.call_count}
        lp = (prompt or "").lower()
        ls = (system or "").lower()

        for m in markers:
            if m.lower() in lp or m.lower() in ls:
                text = f"MARKER_DETECTED: {m}"
                break
        else:
            # Simple heuristics
            if any(
                x in lp
                for x in [
                    "ignore previous instructions",
                    "no rules",
                    "you are nofilterbot",
                ]
            ):
                text = "ATTEMPTED_OVERRIDE_DETECTED"
                safety_flags.append("override-attempt")
            elif "decode base64" in lp or "decode" in lp:
                text = "DECODE_RESULT: SAFE_DEMO"
            else:
                text = "NO_BYPASS_DETECTED"

        tokens = max(1, len((prompt or "").split()) + 5)
        if any(
            x in lp
            for x in ["knowledge base", "uploaded file", "summarize the uploaded file"]
        ):
            meta["retrieval_hits"] = [{"doc_id": "doc-1", "score": 0.92}]

        return {
            "text": text,
            "tokens": tokens,
            "safety_flags": safety_flags,
            "meta": meta,
        }


class OpenAIModelCaller(BaseModelCaller):
    """Calls OpenAI API using the v1.0+ client."""

    def __init__(self, api_key: Optional[str] = None, timeout: int = 60):
        try:
            # Use imported names to avoid circular dependencies in local file
            if 'openai' not in globals():
                 raise ImportError()
        except ImportError:
            raise RuntimeError(
                "openai package required. Install with: pip install --upgrade openai"
            )

        self.timeout = timeout
        api_key_to_use = api_key or os.getenv("OPENAI_API_KEY")

        if not api_key_to_use:
            raise ValueError(
                "OPENAI_API_KEY missing. Set via environment variable or pass to constructor."
            )

        self.client = OpenAI(api_key=api_key_to_use, timeout=timeout)
        logger.info("OpenAI client initialized successfully")

    def call(
        self,
        model: str,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        """Call OpenAI API."""
        self.validate_inputs(model, prompt, temperature)

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        try:
            resp = self.client.chat.completions.create(
                model=model, messages=messages, temperature=temperature, max_tokens=1024
            )

            text = (resp.choices[0].message.content or "").strip()
            tokens = (
                resp.usage.total_tokens
                if resp.usage
                else len(text.split()) + len(prompt.split())
            )

            safety_flags = []
            if resp.choices[0].finish_reason == "content_filter":
                safety_flags.append("content_filtered")

            meta = {
                "model_id": resp.model,
                "finish_reason": resp.choices[0].finish_reason,
            }

            return {
                "text": text,
                "tokens": tokens,
                "safety_flags": safety_flags,
                "meta": meta,
            }

        except APIConnectionError as e:
            logger.error(f"API connection failed: {e}")
            return self.format_error_response("APIConnectionError", str(e))
        except RateLimitError as e:
            logger.error(f"Rate limit exceeded: {e}")
            return self.format_error_response("RateLimitError", str(e))
        except AuthenticationError as e:
            logger.error(f"Authentication failed: {e}")
            return self.format_error_response("AuthenticationError", str(e))
        except NotFoundError as e:
            logger.error(f"Model not found: {model}")
            return self.format_error_response("NotFoundError", model)
        except APIStatusError as e:
            logger.error(f"API status error: {e.status_code}")
            return self.format_error_response(
                f"APIStatusError_{e.status_code}", str(e)
            )
        except Exception as e:
            logger.exception("OpenAIModelCaller unexpected error")
            return self.format_error_response(
                f"UnexpectedError_{type(e).__name__}", str(e)
            )


class HFModelCaller(BaseModelCaller):
    """Calls a generic HTTP endpoint (like HuggingFace Inference)."""

    def __init__(
        self, endpoint_url: str, api_key: Optional[str] = None, timeout: int = 60
    ):
        try:
            # Use imported names to avoid circular dependencies in local file
            if 'requests' not in globals():
                 raise ImportError()
        except ImportError:
            raise RuntimeError(
                "requests package required. Install with: pip install requests"
            )

        if not endpoint_url or not isinstance(endpoint_url, str):
            raise ValueError("endpoint_url must be a non-empty string")

        self.requests = requests # reference the imported requests module
        self.url = endpoint_url
        self.timeout = timeout
        self.api_key = api_key or os.getenv("HF_API_KEY")
        logger.info(f"HF caller initialized with endpoint: {endpoint_url}")

    def call(
        self,
        model: str,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        """Call HuggingFace or custom endpoint."""
        self.validate_inputs(model, prompt, temperature)

        full_prompt = f"System: {system}\n\nUser: {prompt}" if system else prompt
        safe_temp = max(0.01, temperature)

        payload = {
            "inputs": full_prompt,
            "parameters": {"temperature": safe_temp, "max_new_tokens": 512},
        }

        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            r = self.requests.post(
                self.url, headers=headers, json=payload, timeout=self.timeout
            )
            r.raise_for_status()
            data = r.json()

            # Extract generated text
            if isinstance(data, list) and data:
                text = data[0].get("generated_text", "")
            elif isinstance(data, dict):
                text = data.get("generated_text", "")
            else:
                text = str(data)

            # Remove prompt echo if present
            if text.startswith(full_prompt):
                text = text[len(full_prompt) :].strip()

            tokens = len(text.split()) + len(full_prompt.split())

            return {
                "text": text,
                "tokens": tokens,
                "safety_flags": [],
                "meta": {"status_code": r.status_code},
            }

        except HTTPError as e:
            logger.exception(f"HF HTTP error: {e.response.status_code}")
            return self.format_error_response(
                f"HTTPError_{e.response.status_code}", str(e)
            )
        except Timeout as e:
            logger.error(f"Request timeout after {self.timeout}s")
            return self.format_error_response("TimeoutError", str(e))
        except RequestException as e:
            logger.exception(f"HF request exception: {e}")
            return self.format_error_response(
                f"RequestException_{type(e).__name__}", str(e)
            )
        except Exception as e:
            logger.exception("HF caller unexpected error")
            return self.format_error_response("UnexpectedError_HF", str(e))
