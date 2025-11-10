# src/agent_logging.py
import json
import os
import threading
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

from langchain_core.messages import BaseMessage

class AgentLogger:
    """
    Logger thread-safe para fluxos LangGraph: turnos, nós, chamadas LLM e retrieve.
    """

    def __init__(self, log_file: Optional[str] = None) -> None:
        self.LOG_FILE = log_file or os.getenv("LOG_FILE", "agent_llm_calls.txt")
        self._log_lock = threading.Lock()
        self._turn_counters = defaultdict(int)  # por sessão/thread

    # ---------- utilitários internos ----------
    def _session_id(self) -> str:
        # usa thread id como "sessão" para alinhar com RunnableConfig(thread_id)
        return str(threading.get_ident())

    def next_turn(self) -> int:
        sid = self._session_id()
        with self._log_lock:
            self._turn_counters[sid] += 1
            return self._turn_counters[sid]

    @staticmethod
    def _now_iso() -> str:
        return datetime.utcnow().isoformat(timespec="milliseconds") + "Z"

    @staticmethod
    def safe_serialize_messages(msgs: Sequence[BaseMessage]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for m in msgs:
            role = getattr(m, "type", getattr(m, "role", m.__class__.__name__)).lower()
            out.append({
                "role": role,
                "content": str(getattr(m, "content", "")),
            })
        return out

    def _append_log(self, event: Dict[str, Any]) -> None:
        event_base = {
            "ts": self._now_iso(),
            "session_id": self._session_id(),
        }
        line = json.dumps({**event_base, **event}, ensure_ascii=False)
        with self._log_lock:
            with open(self.LOG_FILE, "a", encoding="utf-8") as f:
                f.write(line + "\n")

    def log_turn_start(self, turn_id: int, user_text: str) -> None:
        self._append_log({"type": "turn_start", "turn_id": turn_id, "user_text": user_text})

    def log_turn_end(self, turn_id: int, assistant_text: str) -> None:
        self._append_log({"type": "turn_end", "turn_id": turn_id, "assistant_text": assistant_text})

    def log_node_enter(self, turn_id: int, node_name: str, state_snapshot: Dict[str, Any]) -> None:
        self._append_log({"type": "node_enter", "turn_id": turn_id, "node": node_name, "state": state_snapshot})

    def log_node_exit(self, turn_id: int, node_name: str, result_snapshot: Dict[str, Any]) -> None:
        self._append_log({"type": "node_exit", "turn_id": turn_id, "node": node_name, "result": result_snapshot})

    def log_route_decision(self, turn_id: int, needs_search: bool) -> None:
        self._append_log({"type": "route_decision", "turn_id": turn_id, "needs_search": needs_search})

    def log_llm_call(
        self,
        turn_id: int,
        node_name: str,
        messages_in: Sequence[BaseMessage],
        message_out: BaseMessage,
    ) -> None:
        self._append_log({
            "type": "llm_call",
            "turn_id": turn_id,
            "node": node_name,
            "prompt_messages": self.safe_serialize_messages(messages_in),
            "response_message": {
                "role": getattr(message_out, "type", "ai"),
                "content": str(getattr(message_out, "content", "")),
            },
        })

    def log_retrieve(self, turn_id: int, query: str, docs) -> None:
        """
        Loga a consulta ao retriever e um preview dos documentos retornados.
        """
        previews = []
        for i, d in enumerate(docs or []):
            previews.append({
                "rank": i + 1,
                "preview": (getattr(d, "page_content", "") or "")[:240],
                "metadata": getattr(d, "metadata", {}),
            })
        self._append_log({
            "type": "retrieve",
            "turn_id": turn_id,
            "query": query,
            "results_count": len(docs or []),
            "results_preview": previews,
        })
