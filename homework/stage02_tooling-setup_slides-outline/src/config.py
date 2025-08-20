import os
from dotenv import load_dotenv

def load_env():
    """Load environment variables from .env file."""
    load_dotenv()

def get_key(key: str):
    """Get an environment variable by key, raise if missing."""
    value = os.getenv(key)
    if value is None:
        raise KeyError(f"{key} not found in environment")
    return value