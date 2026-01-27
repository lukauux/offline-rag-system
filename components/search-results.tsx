"use client"

import { Badge } from "@/components/ui/badge"
import { Card } from "@/components/ui/card"
import { Separator } from "@/components/ui/separator"
import { FileText, ImageIcon, Music, Sparkles } from "lucide-react"

interface SearchResult {
  id: number
  text: string
  source: string
  page?: number
  score?: number
  type: string
  metadata?: Record<string, any>
}

interface QueryResponse {
  query: string
  answer: string
  sources: SearchResult[]
  numSources: number
  searchType: string
  timestamp: string
}

interface SearchResultsProps {
  queryResponse: QueryResponse | null
}

export function SearchResults({ queryResponse }: SearchResultsProps) {
  const getIcon = (type: string) => {
    switch (type) {
      case "pdf":
      case "docx":
      case "text":
        return <FileText className="h-4 w-4" />
      case "image":
        return <ImageIcon className="h-4 w-4" />
      case "audio":
        return <Music className="h-4 w-4" />
      default:
        return <FileText className="h-4 w-4" />
    }
  }

  const getTypeColor = (type: string) => {
    switch (type) {
      case "pdf":
        return "bg-chart-1/10 text-chart-1 border-chart-1/20"
      case "docx":
        return "bg-chart-2/10 text-chart-2 border-chart-2/20"
      case "image":
        return "bg-chart-3/10 text-chart-3 border-chart-3/20"
      case "audio":
        return "bg-chart-4/10 text-chart-4 border-chart-4/20"
      default:
        return "bg-muted text-muted-foreground"
    }
  }

  if (!queryResponse) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-center max-w-md">
          <Sparkles className="h-12 w-12 mx-auto mb-4 text-muted-foreground" />
          <p className="text-muted-foreground">Ask a question to search your knowledge base</p>
          <p className="text-xs text-muted-foreground mt-2">
            Using CLIP embeddings for cross-modal semantic search
          </p>
        </div>
      </div>
    )
  }

  const { query, answer, sources, numSources, searchType } = queryResponse

  return (
    <div className="space-y-6">
      <div>
        <div className="flex items-center gap-2 mb-2">
          <h2 className="text-sm font-medium text-muted-foreground">Your Question</h2>
          <Badge variant="outline" className="text-xs">
            {searchType}
          </Badge>
        </div>
        <p className="text-lg font-medium text-foreground">{query}</p>
      </div>

      <Separator />

      <Card className="p-6 bg-gradient-to-br from-accent/5 to-accent/10 border-accent/20">
        <div className="flex items-start gap-3 mb-3">
          <div className="p-2 rounded-lg bg-accent/20">
            <Sparkles className="h-5 w-5 text-accent-foreground" />
          </div>
          <div className="flex-1">
            <h3 className="font-semibold text-foreground mb-1">AI-Generated Answer</h3>
            <p className="text-xs text-muted-foreground">
              Based on {numSources} relevant {numSources === 1 ? "source" : "sources"}
            </p>
          </div>
        </div>
        <p className="text-sm text-foreground leading-relaxed whitespace-pre-wrap">{answer}</p>
      </Card>

      <Separator />

      <div>
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-semibold text-foreground">Source Documents</h2>
          <Badge variant="secondary">
            {numSources} {numSources === 1 ? "source" : "sources"}
          </Badge>
        </div>

        <div className="space-y-3">
          {sources.map((result) => {
            const metadata = result.metadata ?? {}
            return (
              <Card key={result.id} className="p-4 hover:shadow-md transition-shadow">
                <div className="flex items-start gap-3">
                  <div className="flex-shrink-0 mt-1">
                    <div className="p-2 rounded-lg bg-primary/10 text-primary">{getIcon(result.type)}</div>
                  </div>

                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 mb-2 flex-wrap">
                      <Badge className={getTypeColor(result.type)} variant="outline">
                        {result.type.toUpperCase()}
                      </Badge>
                      <span className="text-sm font-medium text-foreground">{result.source}</span>
                      {result.page && <span className="text-xs text-muted-foreground">Page {result.page}</span>}
                      {metadata.start_time_label && (
                        <span className="text-xs text-muted-foreground">{metadata.start_time_label}</span>
                      )}
                      {result.score !== undefined && (
                        <Badge variant="outline" className="text-xs ml-auto">
                          {(result.score * 100).toFixed(0)}% relevance
                        </Badge>
                      )}
                    </div>

                    <p className="text-sm text-foreground leading-relaxed mb-2 whitespace-pre-wrap">
                      {result.text}
                    </p>

                    <div className="flex flex-wrap gap-3 text-xs text-muted-foreground">
                      {metadata.file_path && (
                        <span title={metadata.file_path} className="truncate max-w-xs">
                          {metadata.file_path}
                        </span>
                      )}
                      {metadata.chunk_id && <span>Chunk: {metadata.chunk_id}</span>}
                    </div>
                  </div>
                </div>
              </Card>
            )
          })}
        </div>
      </div>
    </div>
  )
}
