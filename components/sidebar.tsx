"use client"

import { AlertTriangle, CheckCircle2, Loader2, Music, FileText, ImageIcon, Trash2 } from "lucide-react"

import { Button } from "@/components/ui/button"
import { ScrollArea } from "@/components/ui/scroll-area"

export interface UploadedFileItem {
  id: string
  name: string
  type: string
  size: number
  status: "processing" | "indexed" | "error"
  chunks?: number
  message?: string
}

interface SidebarProps {
  uploadedFiles: UploadedFileItem[]
  onClearFiles: () => void
}

export function Sidebar({ uploadedFiles, onClearFiles }: SidebarProps) {
  const getFileIcon = (type: string) => {
    if (type.includes("pdf") || type.includes("document")) {
      return <FileText className="h-4 w-4" />
    }
    if (type.includes("image")) {
      return <ImageIcon className="h-4 w-4" />
    }
    if (type.includes("audio")) {
      return <Music className="h-4 w-4" />
    }
    return <FileText className="h-4 w-4" />
  }

  const formatFileSize = (bytes: number) => {
    if (bytes < 1024) return `${bytes} B`
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
  }

  return (
    <div className="w-80 border-r border-border bg-sidebar flex flex-col">
      <div className="p-4 border-b border-sidebar-border">
        <div className="flex items-center justify-between mb-2">
          <h2 className="font-semibold text-sidebar-foreground">Knowledge Base</h2>
          {uploadedFiles.length > 0 && (
            <Button variant="ghost" size="sm" onClick={onClearFiles} className="h-8 text-xs">
              <Trash2 className="h-3 w-3 mr-1" />
              Clear
            </Button>
          )}
        </div>
        <p className="text-xs text-sidebar-foreground/60">
          {uploadedFiles.length} {uploadedFiles.length === 1 ? "file" : "files"} indexed
        </p>
      </div>

      <ScrollArea className="flex-1">
        <div className="p-4 space-y-2">
          {uploadedFiles.length === 0 ? (
            <div className="text-center py-8 text-sidebar-foreground/40 text-sm">No files uploaded yet</div>
          ) : (
            uploadedFiles.map((file) => (
              <div
                key={file.id}
                className="p-3 rounded-lg bg-sidebar-accent border border-sidebar-border hover:bg-sidebar-accent/80 transition-colors"
              >
                <div className="flex items-start gap-3">
                  <div className="mt-0.5 text-sidebar-foreground/70">{getFileIcon(file.type)}</div>
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium text-sidebar-foreground truncate">{file.name}</p>
                    <p className="text-xs text-sidebar-foreground/60 mt-0.5">{formatFileSize(file.size)}</p>
                    <div className="flex items-center gap-1.5 mt-2 text-xs">
                      {file.status === "processing" && (
                        <>
                          <Loader2 className="h-3 w-3 animate-spin text-primary" />
                          <span className="text-primary">Processing...</span>
                        </>
                      )}
                      {file.status === "indexed" && (
                        <>
                          <CheckCircle2 className="h-3 w-3 text-chart-4" />
                          <span className="text-chart-4">
                            Indexed{file.chunks ? ` (${file.chunks} chunks)` : ""}
                          </span>
                        </>
                      )}
                      {file.status === "error" && (
                        <>
                          <AlertTriangle className="h-3 w-3 text-destructive" />
                          <span className="text-destructive">Failed</span>
                        </>
                      )}
                    </div>
                    {file.status === "error" && file.message && (
                      <p className="text-xs text-destructive/80 mt-1 line-clamp-2">{file.message}</p>
                    )}
                  </div>
                </div>
              </div>
            ))
          )}
        </div>
      </ScrollArea>

      <div className="p-4 border-t border-sidebar-border">
        <div className="text-xs text-sidebar-foreground/60 space-y-1">
          <p>Supported formats:</p>
          <p className="text-sidebar-foreground/40">PDF, DOCX, Images, Audio</p>
        </div>
      </div>
    </div>
  )
}
