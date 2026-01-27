import { NextResponse } from "next/server"

export const runtime = "nodejs"

const RAG_SERVER_URL = process.env.RAG_SERVER_URL ?? "http://127.0.0.1:8000"

export async function GET() {
  try {
    const resp = await fetch(`${RAG_SERVER_URL}/health`)
    const data = await resp.json().catch(() => ({ status: resp.ok ? "ok" : "error" }))
    return NextResponse.json(data, { status: resp.status })
  } catch (e) {
    return NextResponse.json({ status: "error", error: e instanceof Error ? e.message : "Failed" }, { status: 500 })
  }
}

