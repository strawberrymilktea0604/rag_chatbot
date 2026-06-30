from django.shortcuts import render
from httpcore import request
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework import status
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer, CrossEncoder
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, Distance, VectorParams
from django.conf import settings
import os
from backend.utils.qdrant_client import get_qdrant_client
from django.http import JsonResponse
from django.core.files.storage import default_storage
from documents.models import Document
from backend.utils.compute_file_hash import compute_file_hash  # Adjusted the import path to match the project structure
from backend.utils.embeddings import HuggingfaceEmbeddingsModel
import uuid
import logging
import nltk
from transformers import AutoTokenizer
from charset_normalizer import detect

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Adjust log level as needed

# Ensure NLTK resources are downloaded
nltk.download('punkt')

class AskView(APIView):
    def post(self, request):
        from langdetect import detect

        # Step 1: Normalize question
        question = request.data.get("question", "").strip()
        if not question:
            logger.error("No question provided in the request.")
            return Response({"error": "Question is required."}, status=status.HTTP_400_BAD_REQUEST)
        logger.info(f"Received question: {question}")

        # Step 2: Embed the question
        embedding_model = HuggingfaceEmbeddingsModel('all-MiniLM-L6-v2')
        question_embedding = embedding_model.embed_texts([question])[0]
        logger.debug("Question embedding generated successfully.")

        # Step 3: Retrieve from Qdrant
        client = get_qdrant_client()
        collection_name = settings.QDRANT_COLLECTION

        # Get top_k from query parameters or default to 20
        top_k = int(request.query_params.get("top_k", 20))
        logger.info(f"Retrieving top {top_k} chunks from Qdrant.")

        try:
            search_result = client.search(
                collection_name=collection_name,
                query_vector=("default", question_embedding),  # Use named vector format
                limit=top_k,
                with_payload=True,
                score_threshold=0.5
            )
            logger.debug(f"Retrieved {len(search_result)} results from Qdrant.")
            # Log the retrieved results
            for i, result in enumerate(search_result):
                logger.debug(f"Result {i + 1}: ID={result.id}, Score={result.score}, Payload={result.payload}")
        except Exception as e:
            logger.exception("Error during Qdrant search.")
            return Response({"error": f"Error during Qdrant search: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        # DEBUG: show search results
        logger.debug(f"Search results: {[r.payload['text'][:100] + '...' for r in search_result]}")

        # Step 4: Rerank results using a cross-encoder (PhoRanker for Vietnamese)
        reranker = CrossEncoder('itdainb/PhoRanker')
        rerank_inputs = [(question, result.payload['text']) for result in search_result]
        rerank_scores = reranker.predict(rerank_inputs)
        logger.debug("Reranking completed successfully.")

        # Log the rerank inputs
        logger.debug(f"Rerank inputs: {[(question, result.payload['text'][:100]) for result in search_result]}")

        # Log the rerank scores
        logger.debug(f"Rerank scores: {rerank_scores}")

        # Combine results with scores and sort by relevance
        reranked_results = [
            {
                "id": result.id,
                "score": rerank_score,
                "payload": result.payload
            }
            for result, rerank_score in zip(search_result, rerank_scores)
        ]
        reranked_results.sort(key=lambda x: x['score'], reverse=True)

        # Keep top 5–8 results
        top_results = reranked_results[:8]
        logger.info(f"Top {len(top_results)} results selected after reranking.")

        # DEBUG: show reranked results
        logger.debug(f"Reranked results: {[r['payload']['text'][:100] + '...' for r in top_results]}")

        # Log the reranked results
        for i, result in enumerate(reranked_results):
            logger.debug(f"Reranked Result {i + 1}: ID={result['id']}, Score={result['score']}, Payload={result['payload']}")

        # Step 5: Assemble prompt in Vietnamese
        system_message = (
            "Bạn là một trợ lý hữu ích. "
            "Hãy trả lời câu hỏi DỰA TRÊN ngữ cảnh được cung cấp bên dưới. "
            "Nếu ngữ cảnh chứa thông tin liên quan, hãy sử dụng nó để trả lời câu hỏi. "
            "Nếu không tìm thấy câu trả lời trong ngữ cảnh, hãy nói rõ 'Tôi không tìm thấy thông tin trong tài liệu'. "
            "Khi trả lời, hãy trích dẫn nguồn bằng mã tài liệu và số trang (ví dụ: [doc1 tr.12])."
        )

        context_blocks = []
        for result in top_results:
            ref = f"[{result['payload']['document_id']} tr.{result['payload'].get('page', '?')} #{result['id']}]"
            context_blocks.append(f"{ref}\n{result['payload']['text']}\n")
        context_text = "\n".join(context_blocks)

        user_message = f"Câu hỏi: {question}"


        # Step 6: Generate response using a chat/completions model
        from google import generativeai
        google_api_key = os.getenv("GEMINI_API_KEY")  # Retrieve API key from environment variables
        if not google_api_key:
            logger.error("Google API key is not configured.")
            return Response({"error": "Google API key is not configured."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        generativeai.configure(api_key=google_api_key)

        # DEBUG: show context sent to LLM
        logger.debug(f"Context sent to LLM: {context_text[:500]}...")

        try:
            model = generativeai.GenerativeModel('gemini-1.5-flash')
            response_obj = model.generate_content(
                f"{system_message}\n\nNGỮ CẢNH:\n{context_text}\n\nCâu hỏi: {question}\nCâu trả lời:",
                generation_config=generativeai.types.GenerationConfig(
                    temperature=0.2,  # Optional, adjust as needed
                    max_output_tokens=512  # Optional, adjust as needed
                )
            )
            # Validate response
            if not response_obj or not hasattr(response_obj, 'text') or not response_obj.text.strip():
                logger.error("Invalid response from Gemini API.")
                return Response({"error": "Invalid response from Gemini API."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

            response = response_obj.text.strip()
            logger.info("Response generated successfully.")
        except Exception as e:
            logger.exception("Error during response generation.")
            return Response({"error": f"An error occurred while generating the response: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        # Step 7: Map citations to document_id, page, and chunk_id
        citations = [
            {
                "document_id": result['payload']['document_id'],
                "page": result['payload'].get('page', 'N/A'),
                "chunk_id": result['id']
            }
            for result in top_results
        ]
        logger.debug("Citations mapped successfully.")

        return Response({"answer": response, "citations": citations}, status=status.HTTP_200_OK)


class FileUploadView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request, *args, **kwargs):
        # DEBUG: show what the server receives
        print(f"Request headers: {dict(request.headers)}")
        print(f"Request data keys: {list(request.data.keys())}")
        print(f"Request files keys: {list(request.FILES.keys())}")

        file = request.FILES.get('file')
        if not file:
            return Response(
                {"error": "No file uploaded. Make sure you use Body -> form-data and key name 'file' (type: File)."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # Log file details
        print(f"Received file: name={file.name}, content_type={getattr(file, 'content_type', None)}, size={file.size}")

        # Basic validation
        allowed = [
            "application/pdf",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "text/plain",
        ]
        if file.content_type not in allowed:
            return Response(
                {"error": "Unsupported file type", "content_type": file.content_type},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # Save temporarily (for debugging) — adjust to your real save/processing flow
        saved_path = default_storage.save(f"uploads/{file.name}", file)
        print(f"Saved uploaded file to: {saved_path}")

        # Compute hash
        file.seek(0)  # Reset file pointer to the beginning before hashing
        file_hash = compute_file_hash(file)

        # Check for duplicates
        existing_doc = Document.objects.filter(hash=file_hash).first()
        if existing_doc:
            return JsonResponse({
                'document_id': existing_doc.document_id,
                'filename': existing_doc.filename,
                'mimetype': existing_doc.mimetype,
                'size': existing_doc.size,
                'hash': existing_doc.hash,
            })

        # Save file
        file_path = default_storage.save(file.name, file)
        # Ensure the file path is absolute
        if not os.path.isabs(file_path):
            file_path = os.path.join(settings.MEDIA_ROOT, file_path)

        # Save metadata
        document = Document.objects.create(
            document_id=file_hash,
            filename=file.name,
            mimetype=file.content_type,
            size=file.size,
            hash=file_hash,
        )
        chunks = []  # Ensure chunks is always defined
        try:
            if file.content_type == 'application/pdf':
                raw_blocks = self.extract_text_and_metadata_from_pdf(file_path, document.document_id)
            elif file.content_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
                raw_blocks = self.extract_text_and_metadata_from_docx(file_path, document.document_id)
            elif file.content_type == 'text/plain':
                raw_blocks = self.extract_text_and_metadata_from_txt(file_path, document.document_id)
            else:
                raise ValueError("Unsupported file type")

            # Clean and chunk text
            chunks = self.clean_and_chunk_text(raw_blocks)
            print(f"Chunks generated: {chunks}")
            print(f"Number of chunks: {len(chunks)}")

            # Generate embeddings using Hugging Face model
            chunk_texts = [chunk['text'] for chunk in chunks]
            model = HuggingfaceEmbeddingsModel('all-MiniLM-L6-v2')
            embeddings = model.embed_texts(chunk_texts)  # Batch embedding
            print(f"Embeddings generated: {embeddings}")
            print(f"Number of embeddings: {len(embeddings)}")

            # Ensure Qdrant collection exists
            client = get_qdrant_client()
            collection_name = settings.QDRANT_COLLECTION
            try:
                client.get_collection(collection_name)
            except Exception:
                client.create_collection(
                    collection_name=collection_name,
                    vectors_config={
                        "default": VectorParams(
                            size=len(embeddings[0]),
                            distance=Distance.COSINE
                        )
                    }
                )

            # Upload chunks and embeddings to Qdrant
            points = [
                PointStruct(
                    id=str(uuid.uuid4()),  # Generate a valid UUID for each point
                    vector={"default": embedding},  # Use named vector with "default"
                    payload=chunk
                )
                for embedding, chunk in zip(embeddings, chunks)
            ]
            print(f"Points being upserted: {points}")
            print(f"Number of points: {len(points)}")

            response = client.upsert(
                collection_name=collection_name,
                points=points
            )
            print(f"Upsert response: {response}")

            # Update document status and store counts
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
            document.token_count = sum(len(tokenizer.encode(chunk['text'], add_special_tokens=False)) for chunk in chunks)
            document.vector_dim = len(embeddings[0])
            document.save()

            # Clean up the temporary file in a storage-agnostic way
            default_storage.delete(saved_path)

            return JsonResponse({"message": "File uploaded and processed successfully", "document_id": document.document_id}, status=201)

        except Exception as e:
            # Clean up the temporary file in case of an error
            if default_storage.exists(file_path):
                default_storage.delete(file_path)
            return JsonResponse({"error": str(e)}, status=500)

    def extract_text_and_metadata_from_pdf(self, file_path, document_id):
        """Extract text and metadata from a PDF."""
        raw_blocks = []
        with fitz.open(file_path) as pdf:
            for page_number, page in enumerate(pdf, start=1):
                text = page.get_text()
                raw_blocks.append({
                    "text": text,
                    "page": page_number,
                    "document_id": document_id
                })
        return raw_blocks

    def extract_text_and_metadata_from_docx(self, file_path, document_id):
        """Extract text and metadata from a DOCX file."""
        from docx import Document as DocxDocument
        doc = DocxDocument(file_path)
        raw_blocks = []
        for i, paragraph in enumerate(doc.paragraphs, start=1):
            if paragraph.text.strip():  # Skip empty paragraphs
                raw_blocks.append({
                    "text": paragraph.text,
                    "section": f"Paragraph {i}",
                    "document_id": document_id
                })
        return raw_blocks

    def extract_text_and_metadata_from_txt(self, file_path, document_id):
        """Extract text and metadata from a TXT file."""
        raw_blocks = []
        with open(file_path, 'r', encoding='utf-8') as file:
            for i, line in enumerate(file, start=1):
                if line.strip():  # Skip empty lines
                    raw_blocks.append({
                        "text": line.strip(),
                        "section": f"Line {i}",
                        "document_id": document_id
                    })
        return raw_blocks

    def clean_and_chunk_text(self, raw_blocks, target_tokens=400, max_tokens=512, overlap_tokens=100):
        """
        Clean and chunk text into overlapping segments using NLTK for sentence tokenization.
        - Sentence-aware splitting for natural boundaries.
        - Token-based splitting to enforce token limits.
        - Enforces a hard cap of 512 tokens per chunk.
        """
        tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        chunks = []
        chunk_id = 0

        for block in raw_blocks:
            # Normalize whitespace
            text = ' '.join(block['text'].split())

            # Split text into sentences using NLTK
            sentences = nltk.sent_tokenize(text)
            current_chunk = []
            current_token_count = 0

            for sentence in sentences:
                # Tokenize the sentence
                sentence_tokens = tokenizer.encode(sentence, add_special_tokens=False)
                sentence_token_count = len(sentence_tokens)
                
                # Handle very long sentences by breaking them down
                if sentence_token_count > max_tokens:
                    logger.warning(f"Found sentence with {sentence_token_count} tokens (over the {max_tokens} limit). Splitting sentence.")
                    # Split the sentence into smaller parts
                    words = sentence.split()
                    temp_text = ""
                    temp_tokens = []
                    for word in words:
                        word_tokens = tokenizer.encode(word + " ", add_special_tokens=False)
                        if len(temp_tokens) + len(word_tokens) > max_tokens:
                            # Save the current temp_text as a chunk
                            chunks.append({
                                "chunk_id": f"chunk_{chunk_id}",
                                "text": temp_text.strip(),
                                "document_id": block["document_id"],
                                "page": block.get("page"),
                                "section": block.get("section"),
                                "order_index": chunk_id
                            })
                            chunk_id += 1
                            temp_text = word + " "
                            temp_tokens = word_tokens
                        else:
                            temp_text += word + " "
                            temp_tokens.extend(word_tokens)
                    # Add any remaining content
                    if temp_text:
                        current_chunk = [temp_text.strip()]
                        current_token_count = len(tokenizer.encode(temp_text, add_special_tokens=False))
                    
                    continue

                # Check if adding this sentence exceeds the max token limit
                if current_token_count + sentence_token_count > max_tokens:
                    # Finalize the current chunk
                    chunk_text = ' '.join(current_chunk)
                    chunks.append({
                        "chunk_id": f"chunk_{chunk_id}",
                        "text": chunk_text,
                        "document_id": block["document_id"],
                        "page": block.get("page"),
                        "section": block.get("section"),
                        "order_index": chunk_id
                    })
                    chunk_id += 1

                    # Start a new chunk with overlap
                    overlap_start = max(0, len(current_chunk) - overlap_tokens)
                    current_chunk = current_chunk[overlap_start:] + [sentence]
                    current_token_count = sum(len(tokenizer.encode(s, add_special_tokens=False)) for s in current_chunk)
                else:
                    # Add the sentence to the current chunk
                    current_chunk.append(sentence)
                    current_token_count += sentence_token_count

            # Add the last chunk if it contains any text
            if current_chunk:
                chunk_text = ' '.join(current_chunk)
                chunks.append({
                    "chunk_id": f"chunk_{chunk_id}",
                    "text": chunk_text,
                    "document_id": block["document_id"],
                    "page": block.get("page"),
                    "section": block.get("section"),
                    "order_index": chunk_id
                })
                chunk_id += 1

        return chunks

class InitQdrantView(APIView):
    def get(self, request):
        client = get_qdrant_client()

        try:
            client.get_collection(settings.QDRANT_COLLECTION)
            return Response({"message": f"Qdrant collection '{settings.QDRANT_COLLECTION}' already exists."})
        except Exception:
            client.recreate_collection(
                collection_name=settings.QDRANT_COLLECTION,
                vectors_config={
                    "default": VectorParams(
                        size=384,  # Adjust based on your embedding model
                        distance=Distance.COSINE  # Use "COSINE", "EUCLID", or "DOT" based on your use case
                    )
                }
            )
            return Response({"message": f"Qdrant collection '{settings.QDRANT_COLLECTION}' created successfully."})
class DeleteQdrantCollectionView(APIView):
    def post(self, request):
        client = get_qdrant_client()
        try:
            client.delete_collection(settings.QDRANT_COLLECTION)
            return Response({"message": "Collection deleted."})
        except Exception as e:
            return Response({"error": str(e)}, status=500)

