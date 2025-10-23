from abc import ABC, abstractmethod
import asyncio

import tiktoken
import aiohttp
import os
import requests
import json
import loguru
encoder = tiktoken.get_encoding("cl100k_base")

def num_tokens_from_string(string: str) -> int:
    """Returns the number of tokens in a text string."""
    try:
        return len(encoder.encode(string))
    except Exception:
        return 0

class Embeddings(ABC):
    """Interface for embedding models."""

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed search docs."""
        raise NotImplementedError

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Embed query text."""
        raise NotImplementedError

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        """Asynchronous Embed search docs."""
        raise NotImplementedError

    async def aembed_query(self, text: str) -> list[float]:
        """Asynchronous Embed query text."""
        raise NotImplementedError
    
    
class EmbeddingHttp(Embeddings):
    def __init__(self) -> None:
        self.des = "embedding api service"
        self.timeout = aiohttp.ClientTimeout(total=3000)
        
        self.url = os.getenv("SILICONFLOW_BASE_URL_EMBEDDING")
        self.api_key = os.getenv("SILICONFLOW_API_KEY_TEST")
        self.model_name= "BAAI/bge-m3"
        self.headers = {
            "Authorization": "Bearer "+self.api_key,
            'Content-Type': 'application/json'}
    def __str__(self) -> str:
        return self.des
    
    def get_model_info(self):
        response = requests.get(self.url, headers=self.headers)
        # 打印响应内容
        loguru.logger.info(f"Status Code:{response.status_code}")
        return json.loads(response.json)
        
    
    async def asyc_embedding(self,doc):    
        payload = {
            "model": self.model_name,
            "input": doc
        }
        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.post(self.url, headers=self.headers, json=payload) as response:
                    if response.status != 200:
                        loguru.logger.info(f"request error code {response.status}")
                    data = await response.json()
                    if isinstance(data,dict):
                        embed = data['data'][0]["embedding"]
                        return embed
                    else:
                        return []
        except Exception as e:
                loguru.logger.info(f"seed request error: {e}")
    
    def _embedding(self,doc):
        payload = {
            "model": self.model_name,
            "input": doc
        }
        # 发送POST请求
        response = requests.post(self.url, json=payload,headers=self.headers)
        # 打印响应内容
        loguru.logger.info(f"Status Code:{response.status_code}")
        data = response.json()
        if isinstance(data,dict):
            embed = data['data'][0]["embedding"]
            return embed
        else:
            return []
        
    @classmethod
    def embed_documents(cls,doc_list):
        embedding_list = [cls()._embedding(doc) for doc in doc_list]
        return embedding_list

    @classmethod
    def aembed_documents(cls,doc_list):
        embedding_list = [asyncio.run(cls().asyc_embedding(doc)) for doc in doc_list]
        return embedding_list
    
    @classmethod
    def embed_query(cls,query):
        return cls()._embedding(query)
    
    @classmethod
    def aembed_query(cls,query):
        return asyncio.run(cls().asyc_embedding(query))
