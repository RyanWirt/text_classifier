from __future__ import annotations

from pathlib import Path

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.litellm import LiteLLMProvider

from text_classifier.config import Settings


def _load_system_prompt(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def build_agent(settings: Settings) -> Agent:
    provider = LiteLLMProvider(
        api_base=settings.litellm_base_url,
        api_key=settings.litellm_api_key,
    )
    model = OpenAIChatModel(settings.litellm_chat_model, provider=provider)
    return Agent(
        model,
        system_prompt=_load_system_prompt(settings.agent_prompt_file),
        name="dsm-v-review-agent",
    )
