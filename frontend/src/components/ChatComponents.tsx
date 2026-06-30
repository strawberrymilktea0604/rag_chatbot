"use client";

import React from "react";
import { Source } from "@/types";
import styles from "./ChatComponents.module.css";

interface SourcesListProps {
  sources: Source[];
}

/**
 * SourcesList Component
 * Displays citations/sources with expandable accordion
 */
export const SourcesList: React.FC<SourcesListProps> = ({ sources }) => {
  const [isExpanded, setIsExpanded] = React.useState(true);

  if (!sources || sources.length === 0) {
    return null;
  }

  return (
    <div className="mt-4 pt-4 border-t border-gray-700">
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="flex items-center gap-2 text-sm font-semibold text-gray-300 hover:text-white transition-colors"
      >
        <span>
          Sources ({sources.length})
        </span>
        <span className="text-gray-400">
          {isExpanded ? "▼" : "▶"}
        </span>
      </button>

      {isExpanded && (
        <div className="mt-3 space-y-2">
          {sources.map((source, index) => (
            <div
              key={`${source.chunk_id}-${index}`}
              className="p-3 bg-gray-800 hover:bg-gray-700 rounded-lg transition-colors cursor-pointer border border-gray-700 hover:border-gray-600"
            >
              <div className="text-xs font-mono text-gray-500 mb-1">
                {source.chunk_id}
              </div>
              <div className="text-sm text-gray-300">
                <span className="text-gray-500">Document:</span> {source.document_id}
              </div>
              {source.page && source.page !== "N/A" && (
                <div className="text-sm text-gray-300 mt-1">
                  <span className="text-gray-500">Page:</span> {source.page}
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

interface LoadingIndicatorProps {
  phase: "searching" | "generating" | null;
}

/**
 * LoadingIndicator Component
 * Shows different loading states for search and generation phases
 */
export const LoadingIndicator: React.FC<LoadingIndicatorProps> = ({ phase }) => {
  if (!phase) return null;

  return (
    <div className="flex items-center gap-2 text-sm text-gray-400">
      <div className={styles.loadingDots}>
        <span className={styles.dot} />
        <span className={styles.dot} />
        <span className={styles.dot} />
      </div>
      <span>
        {phase === "searching"
          ? "Searching knowledge base..."
          : "Generating answer..."}
      </span>
    </div>
  );
};

interface ChatMessageProps {
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
  isLoading?: boolean;
  loadingPhase?: "searching" | "generating" | null;
  error?: string;
}

/**
 * ChatMessage Component
 * Displays individual messages with proper formatting and sources
 */
export const ChatMessage: React.FC<ChatMessageProps> = ({
  role,
  content,
  sources,
  isLoading,
  loadingPhase,
  error,
}) => {
  const isUser = role === "user";

  return (
    <div className={`flex gap-4 mb-6 ${isUser ? "justify-end" : "justify-start"}`}>
      {!isUser && (
        <div className="w-8 h-8 rounded-lg bg-blue-600 flex items-center justify-center flex-shrink-0">
          <span className="text-white font-bold text-sm">AI</span>
        </div>
      )}

      <div
        className={`max-w-2xl ${
          isUser
            ? "bg-gray-800 text-white rounded-2xl rounded-tr-sm px-4 py-3"
            : "flex-1"
        }`}
      >
        {error ? (
          <div className="text-red-400 text-sm">
            <p className="font-semibold mb-1">Error</p>
            <p>{error}</p>
          </div>
        ) : (
          <>
            {!isUser && (
              <div className="text-gray-300 whitespace-pre-wrap leading-relaxed mb-2">
                {content}
              </div>
            )}
            {isUser && <p className="whitespace-pre-wrap leading-relaxed">{content}</p>}

            {!isUser && isLoading && loadingPhase && (
              <LoadingIndicator phase={loadingPhase} />
            )}

            {!isUser && !isLoading && sources && sources.length > 0 && (
              <SourcesList sources={sources} />
            )}
          </>
        )}
      </div>

      {isUser && (
        <div className="w-8 h-8 rounded-lg bg-gray-700 flex items-center justify-center flex-shrink-0">
          <span className="text-white font-bold text-sm">U</span>
        </div>
      )}
    </div>
  );
};

interface ChatInputProps {
  value: string;
  onChange: (value: string) => void;
  onSubmit: (message: string) => void;
  isLoading: boolean;
  placeholder?: string;
}

/**
 * ChatInput Component
 * Input field for user messages with send button
 */
export const ChatInput: React.FC<ChatInputProps> = ({
  value,
  onChange,
  onSubmit,
  isLoading,
  placeholder = "Ask a question about your documents...",
}) => {
  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (value.trim() && !isLoading) {
      onSubmit(value);
      onChange("");
    }
  };

  return (
    <form onSubmit={handleSubmit} className="flex gap-3">
      <input
        type="text"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
        disabled={isLoading}
        className="flex-1 bg-gray-800 border border-gray-700 rounded-lg px-4 py-3 text-white placeholder-gray-500 focus:outline-none focus:border-blue-500 transition-colors disabled:opacity-50"
      />
      <button
        type="submit"
        disabled={isLoading || !value.trim()}
        className="bg-blue-600 hover:bg-blue-700 disabled:bg-gray-700 disabled:cursor-not-allowed px-6 py-3 rounded-lg font-medium text-white transition-colors"
      >
        Send
      </button>
    </form>
  );
};
