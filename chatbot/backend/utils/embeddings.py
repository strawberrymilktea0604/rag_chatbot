import requests
import os
import logging
import torch
import numpy as np

from transformers import AutoTokenizer, AutoModel
from backend.utils.device import get_device

logger = logging.getLogger(__name__)


def mean_pooling(model_output, attention_mask):
    """
    Perform mean pooling on token embeddings, taking attention mask into account.
    This is the standard pooling method for sentence embeddings.
    """
    token_embeddings = model_output.last_hidden_state  # Use last_hidden_state instead of [0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


class GeminiEmbeddingsModel:
    def __init__(self, model_name, api_url=None, api_key=None):
        """
        Initialize the Gemini Embeddings Model.
        :param model_name: Name of the embedding model (e.g., 'text-embedding-004').
        :param api_url: Base URL of the Gemini API.
        :param api_key: API key for authentication.
        """
        self.model_name = model_name
        self.api_key = os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set.")
        self.api_url = api_url or "https://api.gemini.com/v1/embeddings"

    def embed_texts(self, texts):
        """
        Generate embeddings for a list of texts.
        :param texts: List of strings to embed.
        :return: List of embeddings (List[List[float]]).
        """
        if not texts:
            logger.warning("No texts provided for embedding")
            return []
        
        filtered_texts = [t for t in texts if t and t.strip()]
        if not filtered_texts:
            logger.warning("All texts were empty after filtering")
            return []
        
        if len(filtered_texts) != len(texts):
            logger.warning(f"Filtered out {len(texts) - len(filtered_texts)} empty texts")
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model_name,
            "inputs": filtered_texts
        }

        response = requests.post(self.api_url, json=payload, headers=headers)
        if response.status_code != 200:
            raise Exception(f"Failed to generate embeddings: {response.text}")

        data = response.json()
        return data.get("embeddings", [])


class HuggingfaceEmbeddingsModel:
    def __init__(self, model_name=None):
        """
        Initialize the Hugging Face Embeddings Model using Transformers directly.
        :param model_name: Name of the embedding model. If None, uses EMBEDDING_MODEL env var 
                          or defaults to Vietnamese embedding model.
        """
        if model_name is None:
            model_name = os.getenv("EMBEDDING_MODEL", "dangvantuan/vietnamese-embedding")
        self.model_name = model_name
        
        logger.info(f"Loading tokenizer and model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        
        # Move model to GPU if available (using device utility)
        self.device = get_device()
        self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode
        
        logger.info(f"Model loaded successfully on device: {self.device}")

    def _preprocess_text(self, text):
        """
        Preprocess text to avoid tokenizer issues.
        """
        if not text:
            return ""
        
        # Convert to string if not already
        if not isinstance(text, str):
            text = str(text)
        
        # Remove null bytes and other problematic characters
        text = text.replace('\x00', '')
        
        # Replace problematic unicode characters
        text = text.encode('utf-8', errors='ignore').decode('utf-8')
        
        # Normalize whitespace
        text = ' '.join(text.split())
        
        return text.strip()

    def embed_texts(self, texts):
        """
        Generate embeddings for a list of texts using Transformers.
        :param texts: List of strings to embed.
        :return: List of embeddings (List[List[float]]).
        """
        # Handle single string input
        if isinstance(texts, str):
            texts = [texts]
        
        # Filter out empty texts to prevent errors
        if not texts:
            logger.warning("No texts provided for embedding")
            return []
        
        # Preprocess and filter texts
        processed_texts = []
        for t in texts:
            cleaned = self._preprocess_text(t)
            if cleaned and len(cleaned) > 0:
                processed_texts.append(cleaned)
        
        if not processed_texts:
            logger.warning("All texts were empty after filtering")
            return []
        
        if len(processed_texts) != len(texts):
            logger.warning(f"Filtered out {len(texts) - len(processed_texts)} empty/invalid texts")
        
        try:
            # Process in batches to avoid memory issues
            batch_size = 32
            all_embeddings = []
            
            for i in range(0, len(processed_texts), batch_size):
                batch_texts = processed_texts[i:i + batch_size]
                
                # Tokenize the texts
                encoded_input = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=256,
                    return_tensors='pt'
                )
                
                # Move to device
                encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}
                
                # Generate embeddings
                with torch.no_grad():
                    model_output = self.model(**encoded_input)
                
                # Perform mean pooling
                embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
                
                # Normalize embeddings (L2 normalization for cosine similarity)
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                
                # Convert to list
                batch_embeddings = embeddings.cpu().numpy().tolist()
                all_embeddings.extend(batch_embeddings)
            
            return all_embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
    
    def encode(self, texts, convert_to_numpy=True):
        """
        Encode texts to embeddings. Compatible with SentenceTransformer interface.
        :param texts: List of strings or single string to embed.
        :param convert_to_numpy: If True, return numpy array. If False, return list.
        :return: Embeddings as numpy array or list.
        """
        embeddings = self.embed_texts(texts)
        
        if convert_to_numpy:
            return np.array(embeddings)
        return embeddings