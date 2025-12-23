from typing import Any, List, Optional, Dict
from llama_index.core.llms import (
    CustomLLM,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
)
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.llms.callbacks import llm_completion_callback
from llama_index.core.base.llms.types import ChatMessage, ChatResponse
from openai import OpenAI
import os

class OpenRouterLLM(CustomLLM):
    """
    Custom LLM wrapper for OpenRouter to bypass LlamaIndex OpenAI validation.
    """
    model: str
    api_key: str
    base_url: str = "https://openrouter.ai/api/v1"
    context_window: int = 128000
    
    def __init__(self, model: str, api_key: str):
        super().__init__(model=model, api_key=api_key)
        
    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=self.context_window,
            num_output=4096,
            model_name=self.model,
            is_chat_model=True,
        )

    @property
    def _client(self) -> OpenAI:
        return OpenAI(base_url=self.base_url, api_key=self.api_key)

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            **kwargs
        )
        return CompletionResponse(text=response.choices[0].message.content)

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        # Simple streaming implementation
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            **kwargs
        )
        def gen():
            text = ""
            for chunk in response:
                delta = chunk.choices[0].delta.content or ""
                text += delta
                yield CompletionResponse(text=text, delta=delta)
        return gen()

    def chat(self, messages: List[ChatMessage], **kwargs: Any) -> ChatResponse:
        # Convert LlamaIndex ChatMessages to OpenAI dicts
        openai_msgs = []
        for m in messages:
            content = m.content
            # Handle vision blocks if present (simplified)
            if hasattr(m, 'blocks') and m.blocks:
                content_parts = []
                for block in m.blocks:
                    if block.block_type == "text":
                        content_parts.append({"type": "text", "text": block.text})
                    elif block.block_type == "image":
                        # LlamaIndex stores images in various ways, usually url/path
                        # We need to ensure it's a URL or base64 data URL. 
                        # block.url is AnyUrl, must convert to string for JSON serialization.
                         content_parts.append({
                            "type": "image_url",
                            "image_url": {"url": str(block.url)} 
                        })
                content = content_parts
            
            openai_msgs.append({"role": m.role.value, "content": content})

        response = self._client.chat.completions.create(
            model=self.model,
            messages=openai_msgs,
            **kwargs
        )
        return ChatResponse(
            message=ChatMessage(
                role="assistant", 
                content=response.choices[0].message.content
            )
        )


class OpenRouterEmbedding(BaseEmbedding):
    """
    Custom Embedding wrapper for OpenRouter to bypass LlamaIndex validation.
    """
    model_name: str = "openai/text-embedding-3-small"
    api_key: str
    base_url: str = "https://openrouter.ai/api/v1"

    def __init__(self, api_key: str, model_name: str = "openai/text-embedding-3-small", **kwargs):
        super().__init__(model_name=model_name, api_key=api_key, **kwargs)

    @property
    def _client(self) -> OpenAI:
        return OpenAI(base_url=self.base_url, api_key=self.api_key)

    def _get_query_embedding(self, query: str) -> List[float]:
        return self._get_embedding(query)

    def _get_text_embedding(self, text: str) -> List[float]:
        return self._get_embedding(text)

    def _get_embedding(self, text: str) -> List[float]:
        text = text.replace("\n", " ")
        response = self._client.embeddings.create(
            model=self.model_name,
            input=[text],
            encoding_format="float"
        )
        return response.data[0].embedding

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)
