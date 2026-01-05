"""
Embedding utility module for handling embedding model calls.
Supports OpenAI API compatible embedding services and local embedding models.
"""

import os
from typing import List

from dotenv import load_dotenv
from loguru import logger
from openai import OpenAI

load_dotenv(override=True)

# 配置：使用本地嵌入模型还是API
USE_LOCAL_EMBEDDING = os.getenv("USE_LOCAL_EMBEDDING", "true").lower() in ("1", "true", "yes", "y")
LOCAL_EMBEDDING_MODEL_PATH = os.getenv("LOCAL_EMBEDDING_MODEL_PATH", "")  # 本地嵌入模型路径，通过环境变量设置
LOCAL_EMBEDDING_DEVICE = os.getenv("LOCAL_EMBEDDING_DEVICE", "cuda:1")  # 使用的GPU设备，使用GPU 1，避免与LLM模型（GPU 0）竞争


class EmbeddingModel:
    """
    A wrapper class for embedding model calls using OpenAI API.
    Supports configurable embedding models through environment variables.
    """

    def __init__(self, model: str = None, api_key: str = None, base_url: str = None):
        """
        Initialize the embedding model.

        Args:
            model (str): The embedding model name. If None, reads from EMBEDDING_MODEL env var.
            api_key (str): The API key. If None, reads from EMBEDDING_API_KEY or OPENAI_API_KEY env var.
            base_url (str): The base URL. If None, reads from EMBEDDING_BASE_URL or OPENAI_BASE_URL env var.
        """

        self.model = model or os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")

        # Use dedicated embedding API key if provided, otherwise fall back to OPENAI_API_KEY
        self.api_key = (
            api_key or os.getenv("EMBEDDING_API_KEY") or os.getenv("OPENAI_API_KEY")
        )

        # Use dedicated embedding base URL if provided, otherwise fall back to OPENAI_BASE_URL
        self.base_url = (
            base_url or os.getenv("EMBEDDING_BASE_URL") or os.getenv("OPENAI_BASE_URL")
        )

        if not self.api_key:
            raise ValueError(
                "API key is required. Please set EMBEDDING_API_KEY or OPENAI_API_KEY in .env file."
            )

        # Initialize OpenAI client
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

        logger.info(
            f"Initialized EmbeddingModel with model: {self.model}, base_url: {self.base_url}"
        )
        # {{END MODIFICATIONS}}

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of documents.

        Args:
            texts (List[str]): The list of texts to embed.

        Returns:
            List[List[float]]: The list of embedding vectors.
        """
        if not texts:
            return []

        if USE_LOCAL_EMBEDDING:
            # 使用本地嵌入模型
            from alphasql_rstar.llm_call.local_embedding import embed_texts_local
            try:
                embeddings = embed_texts_local(
                    texts=texts,
                    model_path=LOCAL_EMBEDDING_MODEL_PATH,
                    device=LOCAL_EMBEDDING_DEVICE
                )
                logger.debug(f"Successfully embedded {len(texts)} documents using local model")
                return embeddings
            except Exception as e:
                logger.error(f"Error during local embedding: {str(e)}")
                raise
        else:
            # 使用API
            try:
                # OpenAI API allows batch embedding
                response = self.client.embeddings.create(model=self.model, input=texts)

                # Extract embeddings from response
                embeddings = [data.embedding for data in response.data]

                logger.debug(f"Successfully embedded {len(texts)} documents")
                return embeddings

            except Exception as e:
                logger.error(f"Error during embedding: {str(e)}")
                raise

    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query text.

        Args:
            text (str): The text to embed.

        Returns:
            List[float]: The embedding vector.
        """
        if USE_LOCAL_EMBEDDING:
            # 使用本地嵌入模型
            from alphasql_rstar.llm_call.local_embedding import embed_texts_local
            try:
                embeddings = embed_texts_local(
                    texts=[text],
                    model_path=LOCAL_EMBEDDING_MODEL_PATH,
                    device=LOCAL_EMBEDDING_DEVICE
                )
                logger.debug("Successfully embedded query using local model")
                return embeddings[0]
            except Exception as e:
                logger.error(f"Error during local embedding: {str(e)}")
                raise
        else:
            # 使用API
            try:
                response = self.client.embeddings.create(model=self.model, input=[text])

                embedding = response.data[0].embedding

                logger.debug("Successfully embedded query")
                return embedding

            except Exception as e:
                logger.error(f"Error during embedding: {str(e)}")
                raise


EMBEDDING_MODEL_CALLABLE = None


def get_embedding_model() -> EmbeddingModel:
    """
    Get or create the global embedding model instance.

    Returns:
        EmbeddingModel: The global embedding model instance.
    """

    global EMBEDDING_MODEL_CALLABLE

    if EMBEDDING_MODEL_CALLABLE is None:
        EMBEDDING_MODEL_CALLABLE = EmbeddingModel()

    return EMBEDDING_MODEL_CALLABLE
