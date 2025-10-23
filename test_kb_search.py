from backend.agent.kb_search import KBSimpleSearch
from dotenv import load_dotenv
load_dotenv()
if __name__ == "__main__":
    search = KBSimpleSearch()
    print(search.kb_search(query="高处坠落"))