import os
import requests
import json
from typing import List, Dict, Any
import base64
from io import BytesIO
from PIL import Image
import httpx, asyncio


def _default_options() -> Dict[str, Any]:
    """Default Ollama generation options.

    Ollama's API default is `temperature=0.8` with no seed, which produced
    ~20-30 absolute composite-point swings between same-code, same-query runs
    in the v5 attempted experiments. Setting a low non-zero temperature and a
    fixed seed makes single-run scoring meaningful again without collapsing
    to the v5-d failure mode (greedy decoding + safety-encouraging retry
    prompt = the model always picks "I cannot answer").

    Both knobs are env-overridable so callers can experiment without code
    changes:

      * ``OLLAMA_TEMPERATURE`` (default 0.3) — generation temperature.
      * ``OLLAMA_SEED`` (default 42) — RNG seed for reproducibility.
    """
    return {
        "temperature": float(os.environ.get("OLLAMA_TEMPERATURE", "0.3")),
        "seed": int(os.environ.get("OLLAMA_SEED", "42")),
    }


class OllamaClient:
    """
    An enhanced client for Ollama that now handles image data for VLM models.
    """
    def __init__(self, host: str = "http://localhost:11434"):
        self.host = host
        self.api_url = f"{host}/api"
        # (Connection check remains the same)

    def _image_to_base64(self, image: Image.Image) -> str:
        """Converts a Pillow Image to a base64 string."""
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def generate_embedding(self, model: str, text: str) -> List[float]:
        try:
            response = requests.post(
                f"{self.api_url}/embeddings",
                json={"model": model, "prompt": text}
            )
            response.raise_for_status()
            return response.json().get("embedding", [])
        except requests.exceptions.RequestException as e:
            print(f"Error generating embedding: {e}")
            return []

    def generate_completion(
        self,
        model: str,
        prompt: str,
        *,
        format: str = "",
        images: List[Image.Image] | None = None,
        enable_thinking: bool | None = None,
        options: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """
        Generates a completion, now with optional support for images.

        Args:
            model: The name of the generation model (e.g., 'llava', 'qwen-vl').
            prompt: The text prompt for the model.
            format: The format for the response, e.g., "json".
            images: A list of Pillow Image objects to send to the VLM.
            enable_thinking: Optional flag to disable chain-of-thought for Qwen models.
            options: Optional Ollama ``options`` overrides. Caller-supplied keys
                override the defaults from :func:`_default_options`.
        """
        try:
            merged_options = {**_default_options(), **(options or {})}
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": merged_options,
            }
            if format:
                payload["format"] = format

            if images:
                payload["images"] = [self._image_to_base64(img) for img in images]

            if enable_thinking is not None:
                payload["chat_template_kwargs"] = {"enable_thinking": enable_thinking}

            response = requests.post(
                f"{self.api_url}/generate",
                json=payload
            )
            response.raise_for_status()
            response_lines = response.text.strip().split('\n')
            final_response = json.loads(response_lines[-1])
            return final_response

        except requests.exceptions.RequestException as e:
            print(f"Error generating completion: {e}")
            return {}

    # -------------------------------------------------------------
    # Async variant – uses httpx so the caller can await multiple
    # LLM calls concurrently (triage, verification, etc.).
    # -------------------------------------------------------------
    async def generate_completion_async(
        self,
        model: str,
        prompt: str,
        *,
        format: str = "",
        images: List[Image.Image] | None = None,
        enable_thinking: bool | None = None,
        timeout: int = 60,
        options: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """Asynchronous version of generate_completion using httpx."""

        merged_options = {**_default_options(), **(options or {})}
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": merged_options,
        }
        if format:
            payload["format"] = format
        if images:
            payload["images"] = [self._image_to_base64(img) for img in images]

        if enable_thinking is not None:
            payload["chat_template_kwargs"] = {"enable_thinking": enable_thinking}

        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(f"{self.api_url}/generate", json=payload)
                resp.raise_for_status()
                return json.loads(resp.text.strip().split("\n")[-1])
        except (httpx.HTTPError, asyncio.CancelledError) as e:
            print(f"Async Ollama completion error: {e}")
            return {}

    # -------------------------------------------------------------
    # Streaming variant – yields token chunks in real time
    # -------------------------------------------------------------
    def stream_completion(
        self,
        model: str,
        prompt: str,
        *,
        images: List[Image.Image] | None = None,
        enable_thinking: bool | None = None,
        options: Dict[str, Any] | None = None,
    ):
        """Generator that yields partial *response* strings as they arrive.

        Example:

            for tok in client.stream_completion("qwen2", "Hello"):
                print(tok, end="", flush=True)
        """
        merged_options = {**_default_options(), **(options or {})}
        payload: Dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": True,
            "options": merged_options,
        }
        if images:
            payload["images"] = [self._image_to_base64(img) for img in images]
        if enable_thinking is not None:
            payload["chat_template_kwargs"] = {"enable_thinking": enable_thinking}

        with requests.post(f"{self.api_url}/generate", json=payload, stream=True) as resp:
            resp.raise_for_status()
            for raw_line in resp.iter_lines():
                if not raw_line:
                    # Keep-alive newline
                    continue
                try:
                    data = json.loads(raw_line.decode())
                except json.JSONDecodeError:
                    continue
                # The Ollama streaming API sends objects like {"response":"Hi","done":false}
                chunk = data.get("response", "")
                if chunk:
                    yield chunk
                if data.get("done"):
                    break

if __name__ == '__main__':
    # This test now requires a VLM model like 'llava' or 'qwen-vl' to be pulled.
    print("Ollama client updated for multimodal (VLM) support.")
    try:
        client = OllamaClient()
        # Create a dummy black image for testing
        dummy_image = Image.new('RGB', (100, 100), 'black')
        
        # Test VLM completion
        vlm_response = client.generate_completion(
            model="llava", # Make sure you have run 'ollama pull llava'
            prompt="What color is this image?",
            images=[dummy_image]
        )
        
        if vlm_response and 'response' in vlm_response:
            print("\n--- VLM Test Response ---")
            print(vlm_response['response'])
        else:
            print("\nFailed to get VLM response. Is 'llava' model pulled and running?")

    except Exception as e:
        print(f"An error occurred: {e}")