/**
 * RAG Chatbot Type Definitions
 * Defines TypeScript interfaces for the RAG chatbot application
 */

/**
 * Represents a citation/source for an AI-generated answer
 */
export interface Source {
  document_id: string;
  page: string | number;
  chunk_id: string;
}

/**
 * Represents metadata about the retrieval process
 */
export interface RetrievalMetrics {
  initial_count: number;
  after_filtering: number;
  after_reranking: number;
  passed_min_rerank_score: number;
  min_rerank_score_used: number;
  failed_rerank_threshold: number;
  top_10_selected: number;
  final_top_k: number;
  chunks_filtered_out: number;
}

/**
 * Represents context expansion metadata
 */
export interface ContextExpansion {
  enabled: boolean;
  chunks_added?: number;
  expand_reason?: string;
}

/**
 * Represents debug information from the backend
 */
export interface DebugInfo {
  retrieval_source: string;
  retrieval_metrics: RetrievalMetrics;
  context_expansion: ContextExpansion;
  inference_time_ms?: number;
  embedding_time_ms?: number;
  reranking_time_ms?: number;
  total_latency_ms?: number;
}

/**
 * Represents the API response from the backend
 */
export interface AskResponse {
  answer: string;
  citations: Source[];
  debug?: DebugInfo;
  error?: string;
}

/**
 * Represents a single message in the chat history
 */
export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
  timestamp: Date;
  isLoading?: boolean;
  error?: string;
}

/**
 * Represents the state of the chat application
 */
export interface ChatState {
  messages: Message[];
  isLoading: boolean;
  loadingPhase: "searching" | "generating" | null;
  error: string | null;
}

/**
 * Represents the request payload for the ask endpoint
 */
export interface AskRequest {
  question: string;
  document_id?: string;
  top_k?: number;
}

/**
 * Represents an uploaded document
 */
export interface UploadedDocument {
  id: string;
  file_name: string;
  chunks_count: number;
  document_id: string;
  timestamp: Date;
  status: "uploading" | "success" | "error";
  error?: string;
}

/**
 * Represents the response from the upload endpoint
 */
export interface UploadResponse {
  message: string;
  file_name: string;
  chunks_count: number;
  document_id: string;
  status: string;
  chunk_eval?: any;
  error?: string;
}
