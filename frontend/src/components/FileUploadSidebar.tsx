"use client";

import React, { useState, useRef } from "react";
import { UploadedDocument } from "@/types";
import styles from "./FileUploadSidebar.module.css";
import { Upload, ChevronDown, ChevronUp, Trash2 } from "lucide-react";

interface FileUploadSidebarProps {
  documents: UploadedDocument[];
  onUpload: (file: File) => Promise<void>;
  onDelete: (documentId: string) => void;
  isUploading?: boolean;
}

/**
 * FileUploadSidebar Component
 * Collapsible sidebar for uploading documents and managing uploaded files
 */
export const FileUploadSidebar: React.FC<FileUploadSidebarProps> = ({
  documents,
  onUpload,
  onDelete,
  isUploading = false,
}) => {
  const [isOpen, setIsOpen] = useState(true);
  const [isDragActive, setIsDragActive] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const ALLOWED_EXTENSIONS = new Set([".pdf", ".docx", ".txt"]);

  /**
   * Validate file type
   */
  const validateFile = (file: File): boolean => {
    const fileExtension = `.${file.name.split(".").pop()?.toLowerCase()}`;
    return ALLOWED_EXTENSIONS.has(fileExtension);
  };

  /**
   * Handle file selection (both button click and drag-drop)
   */
  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    console.log("File selected:", file.name, file.size, file.type);

    if (!validateFile(file)) {
      alert(`Only PDF, DOCX, and TXT files are supported. Got: ${file.name}`);
      return;
    }

    try {
      await onUpload(file);
      // Reset input
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
    } catch (error) {
      console.error("Upload error:", error);
    }
  };

  /**
   * Handle button click to trigger file input
   */
  const handleButtonClick = () => {
    fileInputRef.current?.click();
  };

  /**
   * Handle drag over
   */
  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setIsDragActive(true);
    } else if (e.type === "dragleave") {
      setIsDragActive(false);
    }
  };

  /**
   * Handle drop
   */
  const handleDrop = async (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragActive(false);

    const files = e.dataTransfer.files;
    if (files?.[0]) {
      const file = files[0];

      console.log("File dropped:", file.name, file.size, file.type);

      if (!validateFile(file)) {
        alert(`Only PDF, DOCX, and TXT files are supported. Got: ${file.name}`);
        return;
      }

      try {
        await onUpload(file);
      } catch (error) {
        console.error("Upload error:", error);
      }
    }
  };

  return (
    <div className={styles.sidebar}>
      {/* Sidebar Header */}
      <div className={styles.sidebarHeader}>
        <button
          onClick={() => setIsOpen(!isOpen)}
          className={styles.toggleButton}
        >
          <span className={styles.headerTitle}>📁 Documents</span>
          {isOpen ? (
            <ChevronUp size={20} />
          ) : (
            <ChevronDown size={20} />
          )}
        </button>
      </div>

      {/* Sidebar Content */}
      {isOpen && (
        <div className={styles.sidebarContent}>
          {/* Upload Area */}
          <div className={styles.uploadForm}>
            <label
              onDragEnter={handleDrag}
              onDragLeave={handleDrag}
              onDragOver={handleDrag}
              onDrop={handleDrop}
              className={`${styles.dropZone} ${
                isDragActive ? styles.dropZoneActive : ""
              }`}
            >
              <div className={styles.dropZoneContent}>
                <Upload size={24} className={styles.uploadIcon} />
                <p className={styles.dropZoneText}>Drag & drop files here</p>
                <p className={styles.dropZoneSubtext}>or click to browse</p>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".pdf,.docx,.txt"
                  disabled={isUploading}
                  onChange={handleFileChange}
                  className={styles.fileInput}
                  style={{ display: "none" }}
                />
              </div>
            </label>

            <button
              type="button"
              onClick={handleButtonClick}
              disabled={isUploading}
              className={styles.uploadButton}
            >
              {isUploading ? "Uploading..." : "Upload File"}
            </button>
          </div>

          {/* Documents List */}
          <div className={styles.documentsList}>
            <h3 className={styles.documentsTitle}>
              Uploaded Documents ({documents.length})
            </h3>

            {documents.length === 0 ? (
              <p className={styles.emptyState}>No documents uploaded yet</p>
            ) : (
              <div className={styles.documentItems}>
                {documents.map((doc) => (
                  <div
                    key={doc.id}
                    className={`${styles.documentItem} ${styles[doc.status]}`}
                  >
                    <div className={styles.documentInfo}>
                      <p className={styles.documentName} title={doc.file_name}>
                        {doc.file_name}
                      </p>
                      {doc.status === "success" && (
                        <p className={styles.documentMeta}>
                          {doc.chunks_count} chunks
                        </p>
                      )}
                      {doc.status === "error" && doc.error && (
                        <p className={styles.documentError}>{doc.error}</p>
                      )}
                      {doc.status === "uploading" && (
                        <p className={styles.documentMeta}>Uploading...</p>
                      )}
                    </div>

                    {doc.status === "success" && (
                      <button
                        onClick={() => onDelete(doc.document_id)}
                        className={styles.deleteButton}
                        title="Delete document"
                      >
                        <Trash2 size={16} />
                      </button>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* File Format Help */}
          <div className={styles.helpSection}>
            <p className={styles.helpTitle}>Supported formats:</p>
            <ul className={styles.helpList}>
              <li>📄 PDF</li>
              <li>📝 DOCX</li>
              <li>📋 TXT</li>
            </ul>
          </div>
        </div>
      )}
    </div>
  );
};
