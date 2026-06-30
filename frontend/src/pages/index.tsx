"use client";

import React, { useState, useRef, useEffect } from "react";
import { Message, AskRequest, AskResponse, ChatState, UploadedDocument, UploadResponse } from "@/types";
import { ChatMessage, ChatInput } from "@/components/ChatComponents";
import { FileUploadSidebar } from "@/components/FileUploadSidebar";
import styles from "./page.module.css";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api";

/**
 * Generate a unique ID for messages (client-side only)
 */
function generateId(): string {
  return `${Date.now()}-${Math.random().toString(36).substring(2, 11)}`;
}

export default function Home() {
  const [chatState, setChatState] = useState<ChatState>({
    messages: [],
    isLoading: false,
    loadingPhase: null,
    error: null,
  });

  const [input, setInput] = useState("");
  const [documents, setDocuments] = useState<UploadedDocument[]>([]);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadMessage, setUploadMessage] = useState<{ type: "success" | "error"; text: string } | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const [isClient, setIsClient] = useState(false);

  // Ensure component only renders on client
  useEffect(() => {
    setIsClient(true);
  }, []);

  // Auto-scroll to latest message
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [chatState.messages]);

  /**
   * Send a message to the RAG backend
   */
  const sendMessage = async (question: string) => {
    if (!question.trim()) return;

    // Add user message to chat
    const userMessage: Message = {
      id: generateId(),
      role: "user",
      content: question,
      timestamp: new Date(),
    };

    setChatState((prev) => ({
      ...prev,
      messages: [...prev.messages, userMessage],
      isLoading: true,
      loadingPhase: "searching",
      error: null,
    }));

    try {
      // Create request payload
      const payload: AskRequest = {
        question: question,
        top_k: 5,
      };

      // Call backend API
      const response = await fetch(`${API_BASE_URL}/ask/`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        throw new Error(`API error: ${response.statusText}`);
      }

      const data: AskResponse = await response.json();

      if (data.error) {
        throw new Error(data.error);
      }

      // Update loading phase to generating
      setChatState((prev) => ({
        ...prev,
        loadingPhase: "generating",
      }));

      // Simulate a brief delay for better UX (response is already generated)
      await new Promise((resolve) => setTimeout(resolve, 300));

      // Add assistant message with sources
      const assistantMessage: Message = {
        id: generateId(),
        role: "assistant",
        content: data.answer,
        sources: data.citations || [],
        timestamp: new Date(),
      };

      setChatState((prev) => ({
        ...prev,
        messages: [...prev.messages, assistantMessage],
        isLoading: false,
        loadingPhase: null,
        error: null,
      }));
    } catch (error) {
      const errorMessage =
        error instanceof Error ? error.message : "An error occurred";

      // Add error message to chat
      const errorMsg: Message = {
        id: generateId(),
        role: "assistant",
        content: "",
        timestamp: new Date(),
        error: errorMessage,
      };

      setChatState((prev) => ({
        ...prev,
        messages: [...prev.messages, errorMsg],
        isLoading: false,
        loadingPhase: null,
        error: errorMessage,
      }));

      console.error("Error sending message:", error);
    }
  };

  /**
   * Handle input submission
   */
  const handleSubmit = (message: string) => {
    sendMessage(message);
  };

  /**
   * Handle file upload
   */
  const handleFileUpload = async (file: File) => {
    const uploadId = generateId();
    const newUpload: UploadedDocument = {
      id: uploadId,
      file_name: file.name,
      chunks_count: 0,
      document_id: "",
      timestamp: new Date(),
      status: "uploading",
    };

    setDocuments((prev) => [...prev, newUpload]);
    setIsUploading(true);
    setUploadMessage(null);

    try {
      const formData = new FormData();
      formData.append("file", file);

      console.log("Uploading file:", file.name, "size:", file.size, "type:", file.type);
      console.log("Target URL:", `${API_BASE_URL}/upload/?force=true`);

      const response = await fetch(`${API_BASE_URL}/upload/?force=true`, {
        method: "POST",
        body: formData,
        // Browser automatically sets Content-Type with boundary
      });

      console.log("Response status:", response.status);
      console.log("Response headers:", response.headers);

      const responseText = await response.text();
      console.log("Raw response:", responseText);

      let responseData;
      try {
        responseData = JSON.parse(responseText);
      } catch (e) {
        console.error("Failed to parse response as JSON:", e);
        throw new Error(`Invalid response from server: ${responseText.substring(0, 100)}`);
      }

      if (!response.ok) {
        throw new Error(responseData.error || `Upload failed: ${response.statusText}`);
      }

      if (!responseData.document_id) {
        console.warn("Response missing document_id:", responseData);
        throw new Error("Server did not return document ID");
      }

      const data: UploadResponse = responseData;

      setDocuments((prev) =>
        prev.map((doc) =>
          doc.id === uploadId
            ? {
                ...doc,
                status: "success",
                chunks_count: data.chunks_count || 0,
                document_id: data.document_id || "",
              }
            : doc
        )
      );

      console.log(`✓ File ${file.name} uploaded successfully:`, data);

      setUploadMessage({
        type: "success",
        text: `✓ ${file.name} uploaded successfully (${data.chunks_count || 0} chunks)`,
      });
    } catch (error) {
      const errorMessage =
        error instanceof Error ? error.message : "Upload failed";

      console.error("Upload error:", errorMessage);

      setDocuments((prev) =>
        prev.map((doc) =>
          doc.id === uploadId
            ? {
                ...doc,
                status: "error",
                error: errorMessage,
              }
            : doc
        )
      );

      setUploadMessage({
        type: "error",
        text: `✗ ${file.name}: ${errorMessage}`,
      });
    } finally {
      setIsUploading(false);
      setTimeout(() => setUploadMessage(null), 5000);
    }
  };

  /**
   * Handle document deletion
   */
  const handleDeleteDocument = async (documentId: string) => {
    // Removes document from local state and UI
    // Future: Connect to backend delete endpoint for vector DB cleanup
    setDocuments((prev) => prev.filter((doc) => doc.document_id !== documentId));
  };

  if (!isClient) {
    return null; // Prevent server-side rendering
  }

  return (
    <div className={styles.container}>
      {/* File Upload Sidebar */}
      <FileUploadSidebar
        documents={documents}
        onUpload={handleFileUpload}
        onDelete={handleDeleteDocument}
        isUploading={isUploading}
      />

      {/* Main Content Area */}
      <div className={styles.mainContent}>
        {/* Header */}
        <div className={styles.header}>
          <div className={styles.headerContent}>
            <h1 className={styles.title}>RAG Chatbot</h1>
            <p className={styles.subtitle}>
              Ask questions about your documents
            </p>
          </div>

          {/* Upload Message Display */}
          {uploadMessage && (
            <div
              className={`${styles.uploadMessageContainer} ${
                uploadMessage.type === "success"
                  ? styles.uploadMessageSuccess
                  : styles.uploadMessageError
              }`}
            >
              {uploadMessage.text}
            </div>
          )}
        </div>

        {/* Messages Container */}
        <div className={styles.messagesContainer}>
          {chatState.messages.length === 0 ? (
            <div className={styles.emptyState}>
              <div className={styles.emptyIcon}>💬</div>
              <h2 className={styles.emptyTitle}>No messages yet</h2>
              <p className={styles.emptyText}>
                Start by asking a question about your documents. The AI will search
                through your knowledge base and provide answers with citations.
              </p>
            </div>
          ) : (
            <div className={styles.messagesList}>
              {chatState.messages.map((message) => (
                <ChatMessage
                  key={message.id}
                  role={message.role}
                  content={message.content}
                  sources={message.sources}
                  isLoading={
                    chatState.isLoading &&
                    message.id === chatState.messages.at(-1)?.id
                  }
                  loadingPhase={chatState.loadingPhase}
                  error={message.error}
                />
              ))}
              <div ref={messagesEndRef} />
            </div>
          )}
        </div>

        {/* Input Area */}
        <div className={styles.inputContainer}>
          <div className={styles.inputWrapper}>
            <ChatInput
              value={input}
              onChange={setInput}
              onSubmit={handleSubmit}
              isLoading={chatState.isLoading}
              placeholder="Ask a question about your documents..."
            />
            {chatState.error && (
              <p className={styles.errorText}>{chatState.error}</p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
