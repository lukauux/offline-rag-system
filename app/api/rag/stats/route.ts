import { NextResponse } from "next/server"

export const runtime = "nodejs"

const RAG_SERVER_URL = process.env.RAG_SERVER_URL ?? "http://127.0.0.1:8000"

export async function GET() {
  try {
    const resp = await fetch(`${RAG_SERVER_URL}/stats`, { method: "GET" })
    if (!resp.ok) {
      const payload = await resp.json().catch(() => ({}))
      return NextResponse.json(payload || { error: "Failed to fetch stats" }, { status: resp.status })
    }
    const data = await resp.json()
    return NextResponse.json(data)
  } catch (e) {
    return NextResponse.json({ error: e instanceof Error ? e.message : "Failed" }, { status: 500 })
  }
}

