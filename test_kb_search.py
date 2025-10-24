from backend.agent.kb_search import KBSimpleSearch
from dotenv import load_dotenv
load_dotenv()
if __name__ == "__main__":
    search = KBSimpleSearch()
    search.kb_search(query="安徽省有那些安全注意条例")