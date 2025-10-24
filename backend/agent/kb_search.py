import os
from pathlib import Path
from typing import Dict, Iterator, List, Optional
from urllib.parse import unquote
import uuid

from langchain_chroma import Chroma
from datasets import IterableDataset, Features, Value
# from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm
from transformers import AutoTokenizer
import chromadb
from langchain_core.documents import Document
# from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings

from backend.models.splitters.md_text_splitter import chunk_md_by_numbered_subtitles



class KBSimpleSearch():
    def __init__(self):
        self.knowledge_base = self.load_local_markdown_iterable_dataset("data/test_markdown")
        # self.text_splitter = self._init_text_split()
        self.chroma_client = self._init_chroma_client()
        self.collection_name = "kb_dev_md"
    
    def docs_simple_text_split(self):
        source_docs = []
        for doc in self.knowledge_base:
            doc_data = Document(page_content=doc["text"], metadata=doc) 
            source_docs.append(doc_data)
            
        # Split docs and keep only unique ones
        print("Splitting documents...")
        docs_processed = []
        unique_texts = {}
        for doc in tqdm(source_docs):
            new_docs = self.text_splitter.split_documents([doc])
            for new_doc in new_docs:
                if new_doc.page_content not in unique_texts:
                    unique_texts[new_doc.page_content] = True
                    docs_processed.append(new_doc)
        print("doc split chunk: ",len(docs_processed))
        return docs_processed

    def get_embedding(self,text:str):
        import requests

        url = "https://api.siliconflow.cn/v1/embeddings"

        payload = {
            "model": "BAAI/bge-m3",
            "input": text
        }
        api_key = os.getenv("SILICONFLOW_API_KEY_TEST")
        headers = {
            "Authorization": "Bearer "+api_key,
            "Content-Type": "application/json"
        }
        response = requests.post(url, json=payload, headers=headers)

        data = response.json()
        if isinstance(data,dict):
            embed = data['data'][0]["embedding"]
            return embed
        else:
            return []
            
    def _init_chroma_client(self):
        chroma_client = chromadb.HttpClient(host='localhost', port=8000)
        return chroma_client
    
    
    def kb_search(self,query:str,top_k:int):
        query_embed = self.get_embedding(query)
        collection = self.chroma_client.get_collection(self.collection_name)
        results = collection.query(
            query_embeddings=query_embed,
            n_results=top_k
        )
        return results
    
    def build_vector_chroma(self):
        collection = self.chroma_client.list_collections()
        list_collection_name = [name.name for name in collection]
        if collection and self.collection_name not in list_collection_name:
                collection = self.chroma_client.create_collection(name=self.collection_name)
        else:
            print("collection name exsit:",self.collection_name)
            collection = self.chroma_client.get_collection(self.collection_name)
        docs_processed=self.md_text_split()
        documents = []
        embeddings = []
        metadatas = []
        ids = []
        for doc in tqdm(docs_processed):
            id = uuid.uuid4()
            doc_embed = self.get_embedding(doc.page_content)
            documents.append(doc.page_content)
            embeddings.append(doc_embed)
            # del doc.metadata["text"]
            metadatas.append(doc.metadata)
            ids.append(id.hex)
            
        collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas 
        )

    def _init_text_split(self):
        text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            AutoTokenizer.from_pretrained("thenlper/gte-small"),
            chunk_size=200,
            chunk_overlap=20,
            add_start_index=True,
            strip_whitespace=True,
            separators=["\n\n", "\n", ".", " ", ""],
        )
        return text_splitter
    
    def md_text_split(self) -> List[Document]:
        docs=[]
        for doc in self.knowledge_base:
            text = Path(doc['source']).read_text(encoding="utf-8", errors="ignore")
            chunks = chunk_md_by_numbered_subtitles(text,doc["filename"])
            docs +=chunks
        return docs
        
    
    def load_local_markdown_iterable_dataset(
        self,
        root_dir: str,
        **kwargs,
    ) -> IterableDataset:
        features = Features({
            "source": Value("string"),
            "filename": Value("string"),
            "size_bytes": Value("int64"),
            "text": Value("string"),
        })
        return IterableDataset.from_generator(
            lambda: self.iter_markdown_files(root_dir, **kwargs),
            features=features,
        )
    
    def iter_markdown_files(
        self,
        root_dir: str,
        recursive: bool = True,
        enc: str = "utf-8",
        errors: str = "ignore",
        exts: Optional[List[str]] = None,
        max_bytes: Optional[int] = None,
    ) -> Iterator[Dict]:
        exts = exts or [".md", ".markdown"]
        base = Path(root_dir).expanduser().resolve()
        it = base.rglob("*") if recursive else base.glob("*")

        for p in it:
            if p.is_file() and p.suffix.lower() in exts:
                try:
                    size = p.stat().st_size
                    if max_bytes is not None:
                        with open(p, "rb") as f:
                            buf = f.read(max_bytes)
                            text = buf.decode(enc, errors=errors)
                    else:
                        text = p.read_text(encoding=enc, errors=errors)

                    yield {
                        "source": str(p.resolve()),
                        "filename": p.name,
                        "size_bytes": int(size),
                        "text": text,
                    }
                except Exception as e:
                    print(f"[WARN] Skip {p}: {e}")