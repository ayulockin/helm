import os
import requests
from typing import Any, Dict, Optional, List
from abc import ABC

from helm.common.optional_dependencies import handle_module_not_found_error
from helm.common.cache import CacheConfig
from helm.common.request import wrap_request_time, Request, RequestResult, GeneratedOutput, ErrorFlags
from helm.clients.client import CachingClient, truncate_sequence

try:
    import google.generativeai as genai
    from google.protobuf.json_format import MessageToDict
    from google.generativeai.types import HarmCategory, HarmBlockThreshold, GenerationConfig
except ModuleNotFoundError as e:
    handle_module_not_found_error(e, ["google-generativeai"])


class SafetySettingPresets:
    BLOCK_NONE = "block_none"
    DEFAULT = "default"

allowed_categories = [
    HarmCategory.HARM_CATEGORY_HARASSMENT,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
]


def _get_safety_settings_for_preset(safety_settings_preset: Optional[str]):
    """
    Return safety_settings dict based on the preset.
    BLOCK_NONE means do not block any content.
    DEFAULT means no custom settings (None).
    """
    if safety_settings_preset is None or safety_settings_preset == SafetySettingPresets.BLOCK_NONE:
        return {category: HarmBlockThreshold.BLOCK_NONE for category in allowed_categories}
    elif safety_settings_preset == SafetySettingPresets.DEFAULT:
        return None
    else:
        raise ValueError(f"Unknown safety_settings_preset: {safety_settings_preset}")


def _get_model_name_for_request(request: Request) -> str:
    # Any custom logic to map the request model name to an underlying generative model name can be placed here.
    # For now, assume model_engine is a valid model name for Generative AI.
    return request.model_engine


class GoogleGenerativeAIClient(CachingClient, ABC):
    """
    A generic client for Google Generative AI models using the google-generativeai SDK.
    """

    def __init__(
        self,
        cache_config: CacheConfig,
        api_key: str,
        safety_settings_preset: Optional[str] = None,
    ):
        super().__init__(cache_config=cache_config)
        genai.configure(api_key=api_key)
        self.safety_settings_preset = safety_settings_preset
        self.safety_settings = _get_safety_settings_for_preset(safety_settings_preset)

    def make_cache_key_with_safety_settings_preset(self, raw_request: Dict, request: Request) -> Dict:
        """Construct the key for the cache using the raw request and include safety settings preset."""
        if self.safety_settings_preset is not None:
            assert "safety_settings_preset" not in raw_request
            return {
                **CachingClient.make_cache_key(raw_request, request),
                "safety_settings_preset": self.safety_settings_preset,
            }
        else:
            return CachingClient.make_cache_key(raw_request, request)

    def make_request(self, request: Request) -> RequestResult:
        """
        Make a request to a text or chat model using the google-generativeai SDK.
        This example is modeled similar to the VertexAITextClient.
        Adjust as needed for chat or multimodal functionalities.
        """
        model_name = _get_model_name_for_request(request)

        # Construct the generation config
        generation_config = GenerationConfig(
            temperature=request.temperature,
            max_output_tokens=request.max_tokens,
            top_p=request.top_p,
            top_k=request.top_k_per_token,
            # These parameters are supported by google.generativeai if the model supports it
            frequency_penalty=request.frequency_penalty,
            presence_penalty=request.presence_penalty,
            # Stop sequences:
            stop_sequences=request.stop_sequences if request.stop_sequences else None,
            candidate_count=request.num_completions,
        )

        # Prepare raw request for caching
        raw_request = {
            "model": model_name,
            "prompt": request.prompt,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "top_k_per_token": request.top_k_per_token,
            "max_tokens": request.max_tokens,
            "stop_sequences": request.stop_sequences,
            "candidate_count": request.num_completions,
            "presence_penalty": request.presence_penalty,
            "frequency_penalty": request.frequency_penalty,
        }

        cache_key = self.make_cache_key_with_safety_settings_preset(raw_request, request)

        def do_it() -> Dict[str, Any]:
            model = genai.GenerativeModel(model_name=model_name)
            response = model.generate_content(
                request.prompt,
                generation_config=generation_config,
                safety_settings=self.safety_settings,
            )
            return response.to_dict()

        try:
            response, cached = self.cache.get(cache_key, wrap_request_time(do_it))
        except Exception as e:
            error = f"Google GenerativeAIClient error: {e}"
            return RequestResult(success=False, cached=False, error=error, completions=[], embedding=[])
        

        # Extract text from the first candidate
        if "candidates" not in response or not response["candidates"]:
            return RequestResult(success=False, cached=cached, error="No candidates returned", completions=[], embedding=[])

        # Each candidate has 'content' and inside 'parts' with 'text'
        parts = response["candidates"][0]["content"]["parts"]
        response_text = "".join(part.get("text", "") for part in parts)

        # Echo prompt if requested
        text = request.prompt + response_text if request.echo_prompt else response_text

        completion = GeneratedOutput(text=text, logprob=0, tokens=[])
        sequence = truncate_sequence(completion, request, print_warning=True)

        return RequestResult(
            success=True,
            cached=cached,
            request_time=response["request_time"],
            request_datetime=response.get("request_datetime"),
            completions=[sequence],
            embedding=[],
        )
