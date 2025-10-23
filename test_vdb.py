from dotenv import load_dotenv

from backend.vdb.embedding_base import EmbeddingHttp
from backend.vdb.vector_factory import Vector

from langchain_core.documents import Document
load_dotenv()
if __name__ == "__main__":
    # vdb = EmbeddingHttp()
    # print(vdb._embedding(doc="hello world"))
    vdb = Vector(collection_name="test")
    docs =[Document(page_content="hello world")]
    vdb.add_texts(docs)