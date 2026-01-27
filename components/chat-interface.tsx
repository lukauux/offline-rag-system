"use client"

import * as React from "react"
import { useState, useRef, ChangeEvent, FormEvent, KeyboardEvent } from "react"
import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"
import { Send, Loader2, Paperclip } from "lucide-react"

interface Citation {
  id: number;
  source: string;
  type: string;
  page?: number;
  chunk_id: string;
  file_path: string;
  start_time?: number;
  end_time?: number;
  caption?: string;
  ocr_text?: string;
  text: string;
  score: number;
}

interface ChatInterfaceProps {
  onQuery: (query: { text?: string; file?: File; targetModality?: string }) => Promise<{
    success: boolean;
    results: Array<{
      id: number;
      text: string;
      modality: string;
      score: number;
      citation: Citation;
    }>;
  }>;
  isProcessing: boolean;
  disabled?: boolean;
}

export function ChatInterface({ onQuery, isProcessing, disabled }: ChatInterfaceProps) {
  const [query, setQuery] = useState("");
  const [messages, setMessages] = useState<Array<{
    type: "user" | "assistant";
    content: string;
    citations?: Citation[];
  }>>([]);
  const fileInputRef = React.useRef<HTMLInputElement>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if ((query.trim() || fileInputRef.current?.files?.[0]) && !isProcessing && !disabled) {
      const newMessages = [...messages, { type: "user", content: query.trim() }];
      setMessages(newMessages);
      
      try {
        const response = await onQuery({ 
          text: query.trim(),
          file: fileInputRef.current?.files?.[0]
        });
        
        if (response.success) {
          const citations = response.results.map(result => ({
            ...result.citation,
            text: result.text,
            score: result.score
          }));
          
          setMessages([
            ...newMessages,
            {
              type: "assistant",
              content: response.results.map(r => r.text).join("\n\n"),
              citations
            }
          ]);
        }
      } catch (error) {
        console.error("Query failed:", error);
      }
      
      setQuery("");
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      handleSubmit(e)
    }
  }

  return (
    <div className="flex flex-col h-full">
      <div className="flex-grow overflow-auto p-4 space-y-4">
        {messages.map((message, idx) => (
          <div
            key={idx}
            className={`p-4 rounded-lg ${
              message.type === "user" ? "bg-primary/10 ml-12" : "bg-card mr-12"
            }`}
          >
            <div className="prose dark:prose-invert">
              <p>{message.content}</p>
              {message.citations && (
                <div className="mt-2">
                  <p className="text-sm font-semibold">Sources:</p>
                  <ul className="list-none pl-0 space-y-1">
                    {message.citations.map((citation, cidx) => (
                      <li key={cidx} className="text-sm flex items-start gap-2">
                        <span className="text-muted-foreground">[{cidx + 1}]</span>
                        <div>
                          <p className="font-medium">{citation.source}</p>
                          {citation.caption && (
                            <p className="text-muted-foreground">{citation.caption}</p>
                          )}
                          {citation.start_time && citation.end_time && (
                            <p className="text-muted-foreground">
                              {Math.floor(citation.start_time / 60)}:
                              {Math.floor(citation.start_time % 60)
                                .toString()
                                .padStart(2, "0")} -{" "}
                              {Math.floor(citation.end_time / 60)}:
                              {Math.floor(citation.end_time % 60)
                                .toString()
                                .padStart(2, "0")}
                            </p>
                          )}
                        </div>
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          </div>
        ))}
      </div>

      <div className="border-t border-border bg-card p-4">
        <form onSubmit={handleSubmit} className="max-w-4xl mx-auto">
          <div className="flex gap-3">
            <div className="flex-grow flex gap-2">
              <Button
                type="button"
                variant="outline"
                onClick={() => fileInputRef.current?.click()}
                disabled={disabled || isProcessing}
              >
                📎
              </Button>
              <Textarea
                value={query}
                onChange={(e: React.ChangeEvent<HTMLTextAreaElement>) => setQuery(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder={
                  disabled ? "Upload files to start asking questions..." : "Ask a question about your documents..."
                }
                disabled={disabled || isProcessing}
                className="min-h-[60px] max-h-[200px] resize-none bg-background"
              />
            </div>
            <Button
              type="submit"
              size="lg"
              disabled={!query.trim() || isProcessing || disabled}
            >
              {isProcessing ? <Loader2 className="animate-spin" /> : <Send className="h-4 w-4" />}
            </Button>
          </div>
        </form>
      </div>

      <input
        type="file"
        ref={fileInputRef}
        className="hidden"
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (file) {
            handleSubmit(e as unknown as React.FormEvent);
          }
        }}
        accept=".pdf,.docx,.txt,.md,.jpg,.jpeg,.png,.mp3,.wav,.mp4"
      />
    </div>
  )
}
