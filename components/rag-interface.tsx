"use client"

import { useState } from "react"

import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { ChatInterface } from "./chat-interface"
import { FileUploadZone } from "./file-upload-zone"
import { SearchResults } from "./search-results"
import { SearchSettings } from "./search-settings"
import { Sidebar, type UploadedFileItem } from "./sidebar"

interface SearchResult {
  id: number
  text: string
  source: string
  page?: number
  score?: number
  type: string
  metadata?: Record<string, unknown>
}

interface QueryResponse {
  query: string
  answer: string
  sources: SearchResult[]
  numSources: number
  searchType: string
  timestamp: string
}

export function RAGInterface() {
  const [uploadedFiles, setUploadedFiles] = useState<UploadedFileItem[]>([])
  const [queryResponse, setQueryResponse] = useState<QueryResponse | null>(null)
  const [isProcessing, setIsProcessing] = useState(false)
  const [searchType, setSearchType] = useState<"similarity" | "mmr" | "similarity_score">("mmr")
  const [useMultiQuery, setUseMultiQuery] = useState(false)
  const [topK, setTopK] = useState(5)
  const [errorMessage, setErrorMessage] = useState<string | null>(null)

  const updateFileStatus = (id: string, updates: Partial<UploadedFileItem>) => {
    setUploadedFiles((prev) => prev.map((file) => (file.id === id ? { ...file, ...updates } : file)))
  }

  const handleFilesUploaded = async (files: File[]) => {
    setIsProcessing(true)
    setErrorMessage(null)

    for (const file of files) {
      const formData = new FormData()
      formData.append("file", file)

      const tempFile: UploadedFileItem = {
        id: crypto.randomUUID(),
        name: file.name,
        type: file.type,
        size: file.size,
        status: "processing",
      }

      setUploadedFiles((prev) => [...prev, tempFile])

      try {
        const response = await fetch("/api/rag/ingest", {
          method: "POST",
          body: formData,
        })

        const result = await response.json()

        if (!response.ok || !result.success) {
          const message = result.error ?? result.message ?? "Failed to ingest file"
          updateFileStatus(tempFile.id, { status: "error", message })
          setErrorMessage(message)
          continue
        }

        updateFileStatus(tempFile.id, {
          status: "indexed",
          chunks: result.chunks,
          message: result.message,
        })
      } catch (error) {
        console.error("Upload error:", error)
        const message = error instanceof Error ? error.message : "Failed to ingest file"
        updateFileStatus(tempFile.id, { status: "error", message })
        setErrorMessage(message)
      }
    }

    setIsProcessing(false)
  }

  const handleQuery = async (query: string) => {
    setIsProcessing(true)
    setErrorMessage(null)

    try {
      const response = await fetch("/api/rag/query", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          query,
          searchType,
          k: topK,
          useMultiQuery,
        }),
      })

      const result = await response.json()

      if (!response.ok || result.error) {
        const message = result.error ?? "Failed to process query"
        setErrorMessage(message)
        return
      }

      setQueryResponse(result)
    } catch (error) {
      console.error("Query error:", error)
      const message = error instanceof Error ? error.message : "Failed to process query"
      setErrorMessage(message)
    } finally {
      setIsProcessing(false)
    }
  }

  const handleClearFiles = () => {
    setUploadedFiles([])
    setQueryResponse(null)
    setErrorMessage(null)
  }

  return (
    <div className="flex h-screen">
      <Sidebar uploadedFiles={uploadedFiles} onClearFiles={handleClearFiles} />

      <div className="flex-1 flex flex-col">
        <header className="border-b border-border bg-card px-6 py-4">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold text-foreground">Multimodal RAG System</h1>
              <p className="text-sm text-muted-foreground mt-1">
                Offline document intelligence with CLIP + Whisper embeddings
              </p>
            </div>
            <SearchSettings
              searchType={searchType}
              onSearchTypeChange={setSearchType}
              useMultiQuery={useMultiQuery}
              onUseMultiQueryChange={setUseMultiQuery}
              topK={topK}
              onTopKChange={setTopK}
            />
          </div>
          {errorMessage && (
            <Alert variant="destructive" className="mt-4">
              <AlertTitle>Something went wrong</AlertTitle>
              <AlertDescription>{errorMessage}</AlertDescription>
            </Alert>
          )}
        </header>

        <div className="flex-1 flex overflow-hidden">
          <div className="flex-1 flex flex-col">
            <div className="flex-1 overflow-y-auto p-6">
              {uploadedFiles.length === 0 ? (
                <div className="h-full flex items-center justify-center">
                  <FileUploadZone onFilesUploaded={handleFilesUploaded} />
                </div>
              ) : (
                <SearchResults queryResponse={queryResponse} />
              )}
            </div>

            <ChatInterface onQuery={handleQuery} isProcessing={isProcessing} disabled={uploadedFiles.length === 0} />
          </div>
        </div>
      </div>
    </div>
  )
}
