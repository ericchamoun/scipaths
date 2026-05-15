from pathlib import Path
import os

ACL_IDS_PATH = Path("input_ids.json")
PAPERS_DIR = Path("papers")
SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")
