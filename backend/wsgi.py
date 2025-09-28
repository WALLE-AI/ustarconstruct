from dotenv import load_dotenv
load_dotenv()
from app import create_app


app = create_app()

if __name__ == "__main__":
    # For local development
    app.run(host="172.16.40.168", port=8000, debug=False)