# load the hf token
from dotenv import load_dotenv

load_dotenv()
import os

HF_TOKEN = os.getenv("HF_TOKEN")
REPO_ID = os.getenv("REPO_ID")
CACHE_DIR = os.getenv("CACHE_DIR")

from huggingface_hub import snapshot_download

snapshot_download(
    REPO_ID, revision="main", cache_dir=CACHE_DIR, use_auth_token=HF_TOKEN
)