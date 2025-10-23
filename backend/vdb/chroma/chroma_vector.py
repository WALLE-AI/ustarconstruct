import json
import os
from typing import Any

import chromadb
from chromadb import  QueryResult, Settings
from pydantic import BaseModel

from langchain_core.documents import Document
from backend.vdb.embedding_base import Embeddings
from backend.vdb.vector_base import BaseVector
from backend.vdb.vector_factory import AbstractVectorFactory
from backend.vdb.vector_type import VectorType


class ChromaConfig(BaseModel):
    host: str
    port: int
    auth_provider: str | None = None
    auth_credentials: str | None = None
    def to_chroma_params(self):
        settings = Settings(
            # auth
            chroma_client_auth_provider=self.auth_provider,
            chroma_client_auth_credentials=self.auth_credentials,
        )

        return {
            "host": self.host,
            "port": self.port,
            "ssl": False,
            "settings": settings,
        }


class ChromaVector(BaseVector):
    def __init__(self, collection_name: str, config: ChromaConfig):
        super().__init__(collection_name)
        self._client_config = config
        self._client = chromadb.HttpClient(**self._client_config.to_chroma_params())

    def get_type(self) -> str:
        return VectorType.CHROMA

    def create(self, texts: list[Document], embeddings: list[list[float]], **kwargs):
        if texts:
            # create collection
            self.create_collection(self._collection_name)

            self.add_texts(texts, embeddings, **kwargs)

    def create_collection(self, collection_name: str):
        self._client.get_or_create_collection(collection_name)


    def add_texts(self, documents: list[Document], embeddings: list[list[float]], **kwargs):
        uuids = self._get_uuids(documents)
        texts = [d.page_content for d in documents]
        metadatas = [d.metadata for d in documents]

        collection = self._client.get_or_create_collection(self._collection_name)
        # FIXME: chromadb using numpy array, fix the type error later
        collection.upsert(ids=uuids, documents=texts, embeddings=embeddings, metadatas=metadatas)  # type: ignore

    def delete_by_metadata_field(self, key: str, value: str):
        collection = self._client.get_or_create_collection(self._collection_name)
        # FIXME: fix the type error later
        collection.delete(where={key: {"$eq": value}})  # type: ignore

    def delete(self):
        self._client.delete_collection(self._collection_name)

    def delete_by_ids(self, ids: list[str]):
        if not ids:
            return
        collection = self._client.get_or_create_collection(self._collection_name)
        collection.delete(ids=ids)

    def text_exists(self, id: str) -> bool:
        collection = self._client.get_or_create_collection(self._collection_name)
        response = collection.get(ids=[id])
        return len(response) > 0

    def search_by_vector(self, query_vector: list[float], **kwargs: Any) -> list[Document]:
        collection = self._client.get_or_create_collection(self._collection_name)
        document_ids_filter = kwargs.get("document_ids_filter")
        if document_ids_filter:
            results: QueryResult = collection.query(
                query_embeddings=query_vector,
                n_results=kwargs.get("top_k", 4),
                where={"document_id": {"$in": document_ids_filter}},  # type: ignore
            )
        else:
            results: QueryResult = collection.query(query_embeddings=query_vector, n_results=kwargs.get("top_k", 4))  # type: ignore
        score_threshold = float(kwargs.get("score_threshold") or 0.0)

        # Check if results contain data
        if not results["ids"] or not results["documents"] or not results["metadatas"] or not results["distances"]:
            return []

        ids = results["ids"][0]
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        docs = []
        for index in range(len(ids)):
            distance = distances[index]
            metadata = dict(metadatas[index])
            score = 1 - distance
            if score >= score_threshold:
                metadata["score"] = score
                doc = Document(
                    page_content=documents[index],
                    metadata=metadata,
                )
                docs.append(doc)
        # Sort the documents by score in descending order
        docs = sorted(docs, key=lambda x: x.metadata["score"] if x.metadata is not None else 0, reverse=True)
        return docs

    def search_by_full_text(self, query: str, **kwargs: Any) -> list[Document]:
        # chroma does not support BM25 full text searching
        return []


class ChromaVectorFactory(AbstractVectorFactory):
    def init_vector(self, collection_name: str, embeddings: Embeddings) -> BaseVector:
        return ChromaVector(
            collection_name=collection_name,
            config=ChromaConfig(
                host=os.getenv("CHROMA_HOST") or "",
                port=os.getenv("CHROMA_PORT"),
                database=os.getenv("CHROMA_DATABASE") or chromadb.DEFAULT_DATABASE,
                auth_provider=os.getenv("CHROMA_AUTH_PROVIDER") or None,
                auth_credentials=os.getenv("CHROMA_AUTH_CREDENTIALS") or None,
            ),
        )
