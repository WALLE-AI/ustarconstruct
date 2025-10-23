import logging
import os
import time
from abc import ABC, abstractmethod
from typing import Any

# from backend.vdb.cached_embedding import CacheEmbedding
from backend.vdb.embedding_base import EmbeddingHttp, Embeddings
from backend.vdb.vector_base import BaseVector
from backend.vdb.vector_type import VectorType

# from sqlalchemy import select

# from configs import dify_config
# from core.model_manager import ModelManager
# from core.model_runtime.entities.model_entities import ModelType
# from core.rag.datasource.vdb.vector_base import BaseVector
# from core.rag.datasource.vdb.vector_type import VectorType
# from core.rag.embedding.cached_embedding import CacheEmbedding
# from core.rag.embedding.embedding_base import Embeddings
# from core.rag.models.document import Document
# from extensions.ext_database import db
# from extensions.ext_redis import redis_client
# from models.dataset import Dataset, Whitelist
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class AbstractVectorFactory(ABC):
    @abstractmethod
    def init_vector(self, collection_name:str, embeddings: Embeddings) -> BaseVector:
        raise NotImplementedError

    @staticmethod
    def gen_index_struct_dict(vector_type: VectorType, collection_name: str):
        index_struct_dict = {"type": vector_type, "vector_store": {"class_prefix": collection_name}}
        return index_struct_dict


class Vector:
    def __init__(self,collection_name:str,attributes: list | None = None):
        if attributes is None:
            attributes = ["doc_id", "dataset_id", "document_id", "doc_hash"]
        self._embeddings = self._get_embeddings()
        self._attributes = attributes
        self._vector_processor = self._init_vector(collection_name)

    def _init_vector(self,collection_name:str) -> BaseVector:
        vector_type = os.getenv("VECTOR_STORE")

        if not vector_type:
            raise ValueError("Vector store must be specified.")

        vector_factory_cls = self.get_vector_factory(vector_type)
        return vector_factory_cls().init_vector(collection_name, self._embeddings)

    @staticmethod
    def get_vector_factory(vector_type: str) -> type[AbstractVectorFactory]:
        match vector_type:
            case VectorType.CHROMA:
                from  backend.vdb.chroma.chroma_vector import ChromaVectorFactory
                return ChromaVectorFactory
            
            case VectorType.ELASTICSEARCH:
                from backend.vdb.elasticsearch.elasticsearch_vector import ElasticSearchVectorFactory

                return ElasticSearchVectorFactory

            case VectorType.ELASTICSEARCH_JA:
                from backend.vdb.elasticsearch.elasticsearch_ja_vector import (
                    ElasticSearchJaVectorFactory,
                )
                return ElasticSearchJaVectorFactory
            case _:
                raise ValueError(f"Vector store {vector_type} is not supported.")

    def create(self, texts: list | None = None, **kwargs):
        if texts:
            start = time.time()
            logger.info("start embedding %s texts %s", len(texts), start)
            batch_size = 1000
            total_batches = len(texts) + batch_size - 1
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                batch_start = time.time()
                logger.info("Processing batch %s/%s (%s texts)", i // batch_size + 1, total_batches, len(batch))
                batch_embeddings = self._embeddings.embed_documents([document.page_content for document in batch])
                logger.info(
                    "Embedding batch %s/%s took %s s", i // batch_size + 1, total_batches, time.time() - batch_start
                )
                self._vector_processor.create(texts=batch, embeddings=batch_embeddings, **kwargs)
            logger.info("Embedding %s texts took %s s", len(texts), time.time() - start)

    def add_texts(self, documents: list[Document], **kwargs):
        if kwargs.get("duplicate_check", False):
            documents = self._filter_duplicate_texts(documents)

        embeddings = self._embeddings.embed_documents([document.page_content for document in documents])
        self._vector_processor.create(texts=documents, embeddings=embeddings, **kwargs)

    def text_exists(self, id: str) -> bool:
        return self._vector_processor.text_exists(id)

    def delete_by_ids(self, ids: list[str]):
        self._vector_processor.delete_by_ids(ids)

    def delete_by_metadata_field(self, key: str, value: str):
        self._vector_processor.delete_by_metadata_field(key, value)

    def search_by_vector(self, query: str, **kwargs: Any) -> list[Document]:
        query_vector = self._embeddings.embed_query(query)
        return self._vector_processor.search_by_vector(query_vector, **kwargs)

    def search_by_full_text(self, query: str, **kwargs: Any) -> list[Document]:
        return self._vector_processor.search_by_full_text(query, **kwargs)

    def delete(self):
        self._vector_processor.delete()

    def _get_embeddings(self) -> Embeddings:
        return EmbeddingHttp()

    def _filter_duplicate_texts(self, texts: list[Document]) -> list[Document]:
        for text in texts.copy():
            if text.metadata is None:
                continue
            doc_id = text.metadata["doc_id"]
            if doc_id:
                exists_duplicate_node = self.text_exists(doc_id)
                if exists_duplicate_node:
                    texts.remove(text)

        return texts

    def __getattr__(self, name):
        if self._vector_processor is not None:
            method = getattr(self._vector_processor, name)
            if callable(method):
                return method

        raise AttributeError(f"'vector_processor' object has no attribute '{name}'")
