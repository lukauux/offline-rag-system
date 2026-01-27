import { type NextRequest, NextResponse } from "next/server"

export const runtime = "nodejs"

const RAG_SERVER_URL = process.env.RAG_SERVER_URL ?? "http://127.0.0.1:8000"

export async function POST(request: NextRequest) {
  try {
    let searchType = "mmr"
    let k = 5
    let queryText = ""

    // Only support JSON here; map objects to a string if needed
    const body = await request.json().catch(() => ({}))
    const incoming = body?.query
    if (typeof incoming === "string") {
      queryText = incoming
    } else if (incoming && typeof incoming === "object" && typeof incoming.text === "string") {
      queryText = incoming.text
    }

    searchType = body?.searchType ?? searchType
    k = body?.k ?? k

    const backendResponse = await fetch(`${RAG_SERVER_URL}/query`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: queryText, searchType, k }),
    })

    if (!backendResponse.ok) {
      const errorPayload = await backendResponse.json().catch(() => ({}))
      throw new Error(errorPayload.error ?? "Failed to retrieve results")
    }

    const result = await backendResponse.json()

    return NextResponse.json({
      success: true,
      results: result.results,
      totalResults: result.results?.length ?? 0,
      searchType,
      timestamp: new Date().toISOString(),
    })
  } catch (error) {
    console.error("Query error:", error)
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Failed to process query" },
      { status: 500 },
    )
  }
}
