"""
LLM API Call Module - Multi-Provider (Ollama + OpenAI)

This module provides LangChain-based model chains using either:
  - Ollama for local LLM inference (default, used for the local-model runs)
  - OpenAI for cloud LLM inference (used for the GPT-4o reference run)

Set LLM_PROVIDER env var to switch: "ollama" (default) or "openai".

Prerequisites (Ollama):
1. Install Ollama: https://ollama.ai/download
2. Pull required models:
   - ollama pull qwen3:8b           (or your preferred model)
   - ollama pull nomic-embed-text   (for embeddings)
3. pip install langchain-ollama

Prerequisites (OpenAI):
1. pip install langchain-openai
2. Set OPENAI_API_KEY in the environment.

Environment Variables:
- LLM_PROVIDER: "ollama" (default) or "openai"

Ollama-specific:
- OLLAMA_BASE_URL: Ollama server URL (default: http://localhost:11434)
- OLLAMA_MODEL: Chat model name (default: llava-llama3:8b)
- OLLAMA_EMBEDDING_MODEL: Embedding model name (default: nomic-embed-text)
- OLLAMA_TEMPERATURE: Temperature setting (default: 0)
- OLLAMA_NUM_PREDICT: Max tokens to predict (default: 1024)
- OLLAMA_TIMEOUT: Request timeout in seconds (default: 60)

OpenAI-specific:
- OPENAI_MODEL: Chat model name (default: gpt-4o)
- OPENAI_TEMPERATURE: Temperature (default: 0)
- OPENAI_EMBEDDING_MODEL: Embedding model (default: text-embedding-ada-002)

Embeddings:
- EMBEDDING_PROVIDER: "ollama" (default) or "openai"
"""

import os
import time
from dotenv import load_dotenv

# Suppress llama.cpp embedding warnings (harmless but noisy)
os.environ.setdefault('LLAMA_LOG_LEVEL', 'ERROR')

from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from autoe2e.utils import logger
from .utils import log_user_messages


load_dotenv()

# =============================================================================
# Provider Selection
# =============================================================================
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower()  # "ollama" or "openai"

# =============================================================================
# Ollama Configuration
# =============================================================================
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llava-llama3:8b")
OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
OLLAMA_TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", "0"))
OLLAMA_NUM_PREDICT = int(os.getenv("OLLAMA_NUM_PREDICT", "1024"))
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "60"))

# =============================================================================
# OpenAI Configuration
# =============================================================================
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0"))
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002")


# =============================================================================
# Token and Call Tracking
# =============================================================================
class LLMStats:
    """Track LLM usage statistics."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.total_calls = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_llm_time = 0.0  # seconds
        self.embedding_calls = 0
        self.embedding_tokens = 0
        self.call_history = []

    def record_llm_call(self, call_type: str, input_text: str, output_text: str, duration: float):
        """Record an LLM call with estimated token counts (~4 chars per token)."""
        input_tokens = len(input_text) // 4
        output_tokens = len(output_text) // 4

        self.total_calls += 1
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_llm_time += duration

        self.call_history.append({
            'type': call_type,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'duration': duration
        })

    def record_embedding_call(self, texts: list):
        """Record an embedding call."""
        self.embedding_calls += 1
        for text in texts:
            self.embedding_tokens += len(text) // 4

    def get_summary(self) -> dict:
        """Get summary statistics."""
        return {
            'llm_calls': self.total_calls,
            'estimated_input_tokens': self.total_input_tokens,
            'estimated_output_tokens': self.total_output_tokens,
            'estimated_total_tokens': self.total_input_tokens + self.total_output_tokens,
            'total_llm_time_seconds': round(self.total_llm_time, 2),
            'average_call_duration_seconds': round(self.total_llm_time / max(1, self.total_calls), 2),
            'embedding_calls': self.embedding_calls,
            'estimated_embedding_tokens': self.embedding_tokens,
        }


# Global stats tracker
llm_stats = LLMStats()


def create_model_chain(chat_model):
    """
    Create a chain that invokes the model with a system prompt and user messages.

    Returns a function `(system_prompt, user_messages) -> response_text`.
    """
    def invoke_model_chain(system_prompt, user_messages):
        logger.info('Prompt:')
        log_user_messages(user_messages.content)

        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=system_prompt),
            user_messages
        ])
        output_parser = StrOutputParser()

        chain = prompt | chat_model | output_parser

        input_text = system_prompt + str(user_messages.content)

        max_retries = 3
        for attempt in range(max_retries):
            try:
                start_time = time.time()
                res = chain.invoke({})
                duration = time.time() - start_time

                llm_stats.record_llm_call('chat', input_text, res, duration)

                logger.info("Response:")
                logger.info(res)
                logger.info("")
                return res
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"LLM call failed after {max_retries} attempts: {e}")
                    raise
                logger.warn(f"LLM call attempt {attempt + 1} failed: {e}, retrying...")
                time.sleep(1)

    return invoke_model_chain


# =============================================================================
# Initialize Chat Model Based on Provider
# =============================================================================
if LLM_PROVIDER == "openai":
    model = ChatOpenAI(
        model=OPENAI_MODEL,
        temperature=OPENAI_TEMPERATURE,
    )
    logger.info(f"Using OpenAI model: {OPENAI_MODEL}")
else:
    # Default: Ollama
    model = ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=OLLAMA_TEMPERATURE,
        num_predict=OLLAMA_NUM_PREDICT,
        timeout=OLLAMA_TIMEOUT,
    )
    logger.info(f"Using Ollama model: {OLLAMA_MODEL}")

# =============================================================================
# Model Chains
# =============================================================================
# The original AUTOE2E paper used distinct Sonnet/Haiku chains. In this
# implementation a single backend serves both call sites, so the two names
# are kept for source-level compatibility with the paper's call graph.
sonnet_chain = create_model_chain(model)
haiku_chain = create_model_chain(model)

# =============================================================================
# Initialize Embeddings
# =============================================================================
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "ollama").lower()

if EMBEDDING_PROVIDER == "openai":
    embeddings = OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL)
    logger.info(f"Using OpenAI embeddings: {OPENAI_EMBEDDING_MODEL}")
else:
    embeddings = OllamaEmbeddings(
        model=OLLAMA_EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL,
    )
    logger.info(f"Using Ollama embeddings: {OLLAMA_EMBEDDING_MODEL}")

# Backward-compatibility aliases used by infer_utils
ollama_embeddings = embeddings
openai_embeddings = embeddings


# =============================================================================
# Helper Functions
# =============================================================================
def check_ollama_status():
    """Check whether Ollama is running and required models are available."""
    import requests

    status = {
        "ollama_running": False,
        "models_available": [],
        "required_models": [OLLAMA_MODEL, OLLAMA_EMBEDDING_MODEL],
        "missing_models": []
    }

    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        if response.status_code == 200:
            status["ollama_running"] = True
            available = [m["name"] for m in response.json().get("models", [])]
            status["models_available"] = available

            for m in status["required_models"]:
                model_base = m.split(":")[0]
                found = any(model_base in avail for avail in available)
                if not found:
                    status["missing_models"].append(m)
    except requests.exceptions.ConnectionError:
        logger.warn(f"Cannot connect to Ollama at {OLLAMA_BASE_URL}")
    except Exception as e:
        logger.warn(f"Error checking Ollama status: {e}")

    return status


if __name__ == "__main__":
    print("Checking Ollama status...")
    status = check_ollama_status()
    print(f"Ollama running: {status['ollama_running']}")
    print(f"Available models: {status['models_available']}")
    if status['missing_models']:
        print(f"Missing models: {status['missing_models']}")
        print("\nTo pull missing models, run:")
        for m in status['missing_models']:
            print(f"  ollama pull {m}")
