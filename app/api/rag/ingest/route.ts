import { Buffer } from "node:buffer"
import { promises as fs } from "node:fs"
import crypto from "node:crypto"
import path from "node:path"
import { type NextRequest, NextResponse } from "next/server"

export const runtime = "nodejs"

const UPLOAD_DIR = path.join(process.cwd(), ".data", "uploads")
const RAG_SERVER_URL = process.env.RAG_SERVER_URL ?? "http://127.0.0.1:8000"

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData()
    const file = formData.get("file") as File | null

    if (!file) {
      return NextResponse.json({ error: "No file provided" }, { status: 400 })
    }

    // Create upload directory
    await fs.mkdir(UPLOAD_DIR, { recursive: true })

    // Get file type
    const fileType = file.type.split("/")[0] // e.g., "image", "audio", "video"
    const arrayBuffer = await file.arrayBuffer()
    const buffer = Buffer.from(arrayBuffer)

    // Generate unique filename
    const uniquePrefix = crypto.randomUUID().slice(0, 8)
    const sanitizedName = file.name.replace(/[^a-zA-Z0-9._-]/g, "_")
    const storedFilename = `${uniquePrefix}-${sanitizedName}`
    const storedPath = path.join(UPLOAD_DIR, storedFilename)

    // Save file
    await fs.writeFile(storedPath, buffer)

    // Prepare ingest request based on file type
    const ingestRequest: Record<string, any> = {
      filePath: storedPath,
      type: fileType,
    }

    // Add modality-specific parameters
    if (fileType === "image") {
      ingestRequest.ocr = true
      ingestRequest.generateDescription = true
    } else if (fileType === "audio" || fileType === "video") {
      ingestRequest.transcribe = true
    }

    const backendResponse = await fetch(`${RAG_SERVER_URL}/ingest`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(ingestRequest),
    })

    if (!backendResponse.ok) {
      const errorPayload = await backendResponse.json().catch(() => ({}))
      throw new Error(errorPayload.error ?? "Failed to ingest file")
    }

    const result = await backendResponse.json()

    return NextResponse.json({
      success: Boolean(result.success),
      filename: file.name,
      storedPath,
      fileType: fileType,
      size: file.size,
      type: file.type,
      chunks: result.chunks ?? 0,
      message: result.message ?? "File ingested successfully",
    })
  } catch (error) {
    console.error("Ingestion error:", error)
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Failed to ingest file" },
      { status: 500 },
    )
  }
}
