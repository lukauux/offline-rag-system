"""
Lightweight HTTP server that bridges the Next.js frontend with the offline Python RAG pipeline.
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Dict

from .enhanced_rag_pipeline import EnhancedRAGPipeline
from .mistral_rag import MistralRAG
from .config import MISTRAL_API_KEY

PIPELINE = EnhancedRAGPipeline()
MISTRAL = MistralRAG(MISTRAL_API_KEY)


class RAGRequestHandler(BaseHTTPRequestHandler):
    """Handle JSON POST requests for ingest/query/stats operations."""

    def _set_headers(self, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_OPTIONS(self) -> None:  # noqa: N802
        self._set_headers()

    def do_GET(self) -> None:  # noqa: N802
        try:
            if self.path in ("/", "/health"):
                self._set_headers(200)
                self.wfile.write(json.dumps({"status": "ok"}).encode("utf-8"))
                return

            if self.path == "/stats":
                self._set_headers(200)
                self.wfile.write(json.dumps(PIPELINE.get_stats()).encode("utf-8"))
                return

            self._set_headers(404)
            self.wfile.write(json.dumps({"error": "Endpoint not found"}).encode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            self._set_headers(500)
            self.wfile.write(json.dumps({"error": str(exc)}).encode("utf-8"))

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        raw_body = self.rfile.read(length) if length else b"{}"

        try:
            payload = json.loads(raw_body.decode("utf-8"))
        except json.JSONDecodeError:
            self._set_headers(400)
            self.wfile.write(json.dumps({"error": "Invalid JSON payload"}).encode("utf-8"))
            return

        try:
            if self.path == "/query":
                response = self._handle_query(payload)
                self._set_headers(200)
                self.wfile.write(json.dumps(response).encode("utf-8"))
                return

            if self.path == "/ingest":
                response = self._handle_ingest(payload)
                status = 200 if response.get("success") else 400
                self._set_headers(status)
                self.wfile.write(json.dumps(response).encode("utf-8"))
                return

            if self.path == "/stats":
                self._set_headers(200)
                self.wfile.write(json.dumps(PIPELINE.get_stats()).encode("utf-8"))
                return

            self._set_headers(404)
            self.wfile.write(json.dumps({"error": "Endpoint not found"}).encode("utf-8"))

        except Exception as exc:  # pylint: disable=broad-except
            self._set_headers(500)
            self.wfile.write(json.dumps({"error": str(exc)}).encode("utf-8"))

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        # Silence the default HTTP server logging to keep CLI output clean.
        return

    # ------------------------------------------------------------------
    def _handle_query(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        question = payload.get("query", "")
        search_type = payload.get("searchType", "similarity")
        top_k = int(payload.get("k", 5))
        use_multi_query = bool(payload.get("useMultiQuery", False))

        return PIPELINE.query(
            question,
            search_type=search_type,
            k=top_k,
            use_multi_query=use_multi_query,
        )

    # ------------------------------------------------------------------
    def _handle_ingest(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        file_path = payload.get("filePath")
        if isinstance(file_path, str):
            return PIPELINE.ingest_file(file_path)

        files = payload.get("files") or []
        if isinstance(files, list) and files:
            results = []
            for path in files:
                try:
                    result = PIPELINE.ingest_file(path)
                except Exception as exc:  # noqa: BLE001
                    result = {"success": False, "message": str(exc), "file": path}
                results.append(result)

            success_count = sum(1 for item in results if item.get("success"))
            return {
                "success": success_count == len(results),
                "ingested": success_count,
                "results": results,
            }

        return {"success": False, "message": "filePath or files payload is required"}


def run_server(port: int = 8000) -> None:
    server_address = ("", port)
    httpd = HTTPServer(server_address, RAGRequestHandler)
    print(f"Multimodal RAG server listening on http://127.0.0.1:{port}")
    print("Available endpoints: POST /ingest, POST /query, POST /stats")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down RAG server...")
        PIPELINE.save_state()
        httpd.shutdown()


if __name__ == "__main__":
    run_server()
