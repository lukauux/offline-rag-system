"""
Answer generation module that optionally uses a local LLM via llama.cpp.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence

try:  # Optional dependency – only required when a GGUF model path is provided.
    from llama_cpp import Llama  # type: ignore
except ImportError:  # pragma: no cover - handled gracefully at runtime
    Llama = None  # type: ignore


class AnswerEngine:
    """Generate grounded answers from retrieved sources, with LLM fallback."""

    def __init__(self, config: Dict[str, Any]) -> None:
        llm_config = config.get("llm", {})
        self.llm_config = llm_config
        self.llm = None
        self.max_context_chunks = int(llm_config.get("max_context_chunks", 6))
        self.stop_sequences = llm_config.get("stop", ["</s>", "###", "</answer>"])

        model_path = llm_config.get("model_path")
        if model_path:
            if Llama is None:
                raise RuntimeError(
                    "llama-cpp-python is not installed. Install it or remove the 'model_path' from llm config."
                )

            resolved_path = Path(model_path).expanduser().resolve()
            if not resolved_path.exists():
                raise FileNotFoundError(f"LLM model file not found: {resolved_path}")

            kwargs: Dict[str, Any] = {
                "model_path": str(resolved_path),
                "n_ctx": int(llm_config.get("context_window", 4096)),
            }

            if llm_config.get("threads") is not None:
                kwargs["n_threads"] = int(llm_config["threads"])
            if llm_config.get("gpu_layers") is not None:
                kwargs["n_gpu_layers"] = int(llm_config["gpu_layers"])
            if llm_config.get("batch_size") is not None:
                kwargs["n_batch"] = int(llm_config["batch_size"])
            if llm_config.get("use_mlock") is not None:
                kwargs["use_mlock"] = bool(llm_config["use_mlock"])
            if llm_config.get("use_mmap") is not None:
                kwargs["use_mmap"] = bool(llm_config["use_mmap"])

            print(f"Loading local LLM from {resolved_path}...")
            self.llm = Llama(**kwargs)
            print("Local LLM ready for answer generation.")

    # ------------------------------------------------------------------
    def generate(self, question: str, sources: Sequence[Dict[str, Any]]) -> str:
        if not sources:
            return "I could not find any relevant information in the indexed files."

        if not self.llm:
            return self._fallback_summary(question, sources)

        prompt = self._build_prompt(question, sources)
        try:
            response = self.llm.create_completion(
                prompt=prompt,
                max_tokens=int(self.llm_config.get("max_tokens", 384)),
                temperature=float(self.llm_config.get("temperature", 0.1)),
                top_p=float(self.llm_config.get("top_p", 0.95)),
                stop=self.stop_sequences,
            )
            answer = response["choices"][0]["text"].strip()
            if not answer:
                return self._fallback_summary(question, sources)
            return answer
        except Exception as exc:  # pragma: no cover - defensive fallback
            print(f"LLM generation failed: {exc}")
            return self._fallback_summary(question, sources)

    # ------------------------------------------------------------------
    def _build_prompt(self, question: str, sources: Sequence[Dict[str, Any]]) -> str:
        context_blocks: List[str] = []
        for idx, source in enumerate(sources[: self.max_context_chunks], start=1):
            citation = self._format_citation(source.get("metadata", {}))
            text = source.get("text", "").strip().replace("\n", " ")
            context_blocks.append(f"[{idx}] {text}\nSource: {citation}")

        system_prompt = self.llm_config.get(
            "system_prompt",
            "You are an offline assistant. Answer the question using only the provided sources."
            " Cite sources in square brackets (e.g., [1]) and do not fabricate information.",
        )

        context_text = "\n\n".join(context_blocks)

        return (
            f"{system_prompt}\n\n"
            f"Context:\n{context_text}\n\n"
            f"Question: {question}\n"
            f"Answer (include citations like [1]):"
        )

    # ------------------------------------------------------------------
    def _fallback_summary(self, question: str, sources: Sequence[Dict[str, Any]]) -> str:
        highlights: List[str] = []
        for idx, source in enumerate(sources[:3], start=1):
            text = source.get("text", "").strip()
            if not text:
                continue
            citation = self._format_citation(source.get("metadata", {}))
            highlights.append(f"[{idx}] {text} ({citation})")

        lines = ["Based on the retrieved evidence:", *highlights]
        lines.append(f"These results are the closest matches for: '{question}'.")
        if not highlights:
            lines = [
                "I indexed the uploaded documents but could not extract any text to answer that question.",
            ]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    @staticmethod
    def _format_citation(metadata: Dict[str, Any]) -> str:
        source = metadata.get("source", "unknown")
        doc_type = metadata.get("type")

        if doc_type == "pdf" and metadata.get("page"):
            return f"{source}, page {metadata['page']}"
        if doc_type == "audio" and metadata.get("start_time") is not None:
            minutes = int(metadata.get("start_time", 0.0) // 60)
            seconds = int(metadata.get("start_time", 0.0) % 60)
            return f"{source}, {minutes:02d}:{seconds:02d}"
        return source


__all__ = ["AnswerEngine"]
