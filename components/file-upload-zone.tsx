"use client"

import type React from "react"

import { useCallback, useRef, useState } from "react"
import { Upload, FileText, Image, Music } from "lucide-react"
import { Button } from "@/components/ui/button"

interface FileUploadZoneProps {
  onFilesUploaded: (files: File[]) => void
}

export function FileUploadZone({ onFilesUploaded }: FileUploadZoneProps) {
  const [isDragging, setIsDragging] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault()
      setIsDragging(false)

      const files = Array.from(e.dataTransfer.files)
      if (files.length > 0) {
        onFilesUploaded(files)
      }
    },
    [onFilesUploaded],
  )

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(true)
  }, [])

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)
  }, [])

  const handleFileSelect = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const files = Array.from(e.target.files || [])
      if (files.length > 0) {
        onFilesUploaded(files)
      }
    },
    [onFilesUploaded],
  )

  return (
    <div className="w-full max-w-2xl mx-auto">
      <div
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        className={`
          relative border-2 border-dashed rounded-xl p-12 text-center transition-all
          ${
            isDragging
              ? "border-primary bg-primary/5 scale-[1.02]"
              : "border-border bg-card hover:border-primary/50 hover:bg-accent/5"
          }
        `}
      >
        {/* Full-surface invisible input to ensure clicks always open file dialog */}
        <input
          ref={inputRef}
          type="file"
          multiple
          accept=".pdf,.doc,.docx,.txt,.jpg,.jpeg,.png,.webp,.mp3,.wav,.m4a,.ogg"
          onChange={handleFileSelect}
          className="absolute inset-0 opacity-0 cursor-pointer"
          aria-label="Upload files"
        />
        <div className="flex flex-col items-center gap-4">
          <div className="p-4 rounded-full bg-primary/10">
            <Upload className="h-8 w-8 text-primary" />
          </div>

          <div>
            <h3 className="text-xl font-semibold text-foreground mb-2">Upload your files</h3>
            <p className="text-sm text-muted-foreground max-w-md">
              Drag and drop files here, or click to browse. Supports PDF, DOCX, images, and audio files.
            </p>
          </div>

          <div className="flex items-center gap-6 mt-2">
            <div className="flex items-center gap-2 text-xs text-muted-foreground">
              <FileText className="h-4 w-4" />
              <span>Documents</span>
            </div>
            <div className="flex items-center gap-2 text-xs text-muted-foreground">
              <Image className="h-4 w-4" />
              <span>Images</span>
            </div>
            <div className="flex items-center gap-2 text-xs text-muted-foreground">
              <Music className="h-4 w-4" />
              <span>Audio</span>
            </div>
          </div>

          <Button
            size="lg"
            className="mt-4 bg-accent hover:bg-accent/90 text-accent-foreground"
            onClick={() => inputRef.current?.click()}
          >
            Browse Files
          </Button>
        </div>
      </div>
    </div>
  )
}
