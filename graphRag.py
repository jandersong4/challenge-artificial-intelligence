import threading
from collections.abc import Sequence
from typing import Annotated, TypedDict, Literal, Optional, List, Dict, Any
from collections import defaultdict
from datetime import datetime
import json
import os
from src.prompt import classifier_prompt, general_system_prompt
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph, add_messages
from langgraph.graph.state import RunnableConfig
from langchain_pinecone import PineconeVectorStore
from src.helper import downloald_hugging_face_embeddings

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")
os.environ["PINECONE_API_KEY"] = os.getenv("PINECONE_API_KEY", "")

GPT_MODEL = "openai:gpt-4.1-nano"
PINECONE_INDEX_NAME = "chatbot"

llm = init_chat_model(GPT_MODEL)

embeddings = downloald_hugging_face_embeddings()
docsearch = PineconeVectorStore.from_existing_index(index_name=PINECONE_INDEX_NAME, embedding=embeddings)
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 2})

# -------------------- LOG infra --------------------
LOG_FILE = os.getenv("LOG_FILE", "agent_llm_calls.txt")
_log_lock = threading.Lock()
_turn_counters = defaultdict(int)  # por thread/session

def _session_id() -> str:
    # usa thread id como sessão (mantém alinhado ao seu checkpointer/config)
    return str(threading.get_ident())

def _next_turn() -> int:
    sid = _session_id()
    with _log_lock:
        _turn_counters[sid] += 1
        return _turn_counters[sid]

def _now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="milliseconds") + "Z"

def _safe_serialize_messages(msgs: Sequence[BaseMessage]) -> List[Dict[str, Any]]:
    out = []
    for m in msgs:
        role = getattr(m, "type", getattr(m, "role", m.__class__.__name__)).lower()
        out.append({
            "role": role,
            "content": str(getattr(m, "content", "")),
        })
    return out

def _append_log(event: Dict[str, Any]) -> None:
    event_base = {
        "ts": _now_iso(),
        "session_id": _session_id(),
    }
    line = json.dumps({**event_base, **event}, ensure_ascii=False)
    with _log_lock:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")

def log_turn_start(turn_id: int, user_text: str) -> None:
    _append_log({"type": "turn_start", "turn_id": turn_id, "user_text": user_text})

def log_turn_end(turn_id: int, assistant_text: str) -> None:
    _append_log({"type": "turn_end", "turn_id": turn_id, "assistant_text": assistant_text})

def log_node_enter(turn_id: int, node_name: str, state_snapshot: Dict[str, Any]) -> None:
    _append_log({"type": "node_enter", "turn_id": turn_id, "node": node_name, "state": state_snapshot})

def log_node_exit(turn_id: int, node_name: str, result_snapshot: Dict[str, Any]) -> None:
    _append_log({"type": "node_exit", "turn_id": turn_id, "node": node_name, "result": result_snapshot})

def log_route_decision(turn_id: int, needs_search: bool) -> None:
    _append_log({"type": "route_decision", "turn_id": turn_id, "needs_search": needs_search})

def log_llm_call(turn_id: int, node_name: str, messages_in: Sequence[BaseMessage], message_out: BaseMessage) -> None:
    _append_log({
        "type": "llm_call",
        "turn_id": turn_id,
        "node": node_name,
        "prompt_messages": _safe_serialize_messages(messages_in),
        "response_message": {
            "role": getattr(message_out, "type", "ai"),
            "content": str(getattr(message_out, "content", "")),
        },
    })
    
def log_retrieve(turn_id: int, query: str, docs) -> None:
    """Loga a consulta do retriever e um preview dos documentos retornados."""
    previews = []
    for i, d in enumerate(docs or []):
        previews.append({
            "rank": i + 1,
            "preview": (getattr(d, "page_content", "") or "")[:240],
            "metadata": getattr(d, "metadata", {}),
        })
    _append_log({
        "type": "retrieve",
        "turn_id": turn_id,
        "query": query,
        "results_count": len(docs or []),
        "results_preview": previews,
    })


# -------------------- Estado do agente --------------------
class AgentState(TypedDict, total=False):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    needs_search: bool
    context_chunks: Optional[List[str]]
    turn_id: Optional[int]

# -------------------- Helpers --------------------
def _thread_config() -> RunnableConfig:
    return RunnableConfig(configurable={"thread_id": threading.get_ident()})

def _last_user_text(messages: Sequence[BaseMessage]) -> str:
    return str(messages[-1].content) if messages else ""

def _state_snapshot_for_log(state: AgentState) -> Dict[str, Any]:
    # snapshot leve para log (sem documentos grandes)
    snap: Dict[str, Any] = {}
    if "needs_search" in state:
        snap["needs_search"] = state["needs_search"]
    if "context_chunks" in state:
        snap["context_chunks_len"] = len(state.get("context_chunks") or [])
    if "messages" in state:
        # apenas o último para não crescer demais
        msgs = state.get("messages") or []
        snap["last_message"] = _safe_serialize_messages(msgs[-1:])  # último
        snap["messages_count"] = len(msgs)
    if "turn_id" in state and state["turn_id"] is not None:
        snap["turn_id"] = state["turn_id"]
    return snap

# -------------------- Nós --------------------
def classify_need_search(state: AgentState) -> AgentState:
    turn_id = state.get("turn_id") or 0
    log_node_enter(turn_id, "classify", _state_snapshot_for_log(state))

    user_utterance = _last_user_text(state["messages"])
    sys = SystemMessage(content=classifier_prompt.format(context=user_utterance))
    msgs = [sys, HumanMessage(content=user_utterance)]
    judge = llm.invoke(msgs)

    log_llm_call(turn_id, "classify", msgs, judge)

    decision_text = (judge.content or "").strip().upper()
    needs = decision_text.startswith("Y")

    log_route_decision(turn_id, needs)

    result = {"needs_search": needs}
    log_node_exit(turn_id, "classify", result)
    return result

def retrieve_docs(state: AgentState) -> AgentState:
    turn_id = state.get("turn_id") or 0
    log_node_enter(turn_id, "retrieve", _state_snapshot_for_log(state))

    user_utterance = _last_user_text(state["messages"])
    docs = retriever.invoke(user_utterance) 
    chunks = [d.page_content for d in docs]
    
    log_retrieve(turn_id, user_utterance, docs)
    log_node_exit(turn_id, "retrieve",{"docs_count": len(docs), "chunks_len": len(chunks)})
    
    return {"context_chunks": chunks}

def answer_with_context(state: AgentState) -> AgentState:
    turn_id = state.get("turn_id") or 0
    log_node_enter(turn_id, "answer_with_context", _state_snapshot_for_log(state))

    user_utterance = _last_user_text(state["messages"])
    context = "\n\n".join(state.get("context_chunks") or [])
    sys = SystemMessage(content=(
        general_system_prompt
        + "\n\nUse APENAS as informações a seguir como contexto quando forem relevantes."
        + " Se o contexto não contiver a resposta, seja honesto."
        + f"\n\n[CONTEXT]\n{context}\n[/CONTEXT]"
    ))
    msgs = [sys] + state.get("messages", [])  # mantém histórico completo
    resp = llm.invoke(msgs)

    log_llm_call(turn_id, "answer_with_context", msgs, resp)

    result = {"messages": [resp]}
    log_node_exit(turn_id, "answer_with_context", {"assistant_preview": str(resp.content)[:200]})
    return result

def answer_direct(state: AgentState) -> AgentState:
    turn_id = state.get("turn_id") or 0
    log_node_enter(turn_id, "answer_direct", _state_snapshot_for_log(state))

    sys = SystemMessage(content="Você é um assistente útil, conciso e objetivo.")
    msgs = [sys] + state.get("messages", [])
    resp = llm.invoke(msgs)

    # log da chamada de LLM
    log_llm_call(turn_id, "answer_direct", msgs, resp)

    result = {"messages": [resp]}
    log_node_exit(turn_id, "answer_direct", {"assistant_preview": str(resp.content)[:200]})
    return result

# -------------------- Grafo --------------------
builder = StateGraph(AgentState, context_schema=None, input_schema=AgentState, output_schema=AgentState)

builder.add_node("classify", classify_need_search)
builder.add_node("retrieve", retrieve_docs)
builder.add_node("answer_with_context", answer_with_context)
builder.add_node("answer_direct", answer_direct)

builder.add_edge(START, "classify")

def route_after_classify(state: AgentState) -> Literal["retrieve", "answer_direct"]:
    return "retrieve" if state.get("needs_search") else "answer_direct"

builder.add_conditional_edges("classify", route_after_classify)
builder.add_edge("retrieve", "answer_with_context")
builder.add_edge("answer_with_context", END)
builder.add_edge("answer_direct", END)

checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)

# -------------------- API de uso (para sua rota POST) --------------------
def agentic_reply(user_text: str) -> str:
    """
    Recebe a mensagem do usuário e retorna a resposta, mantendo contexto entre turnos.
    Além disso, registra logs completos do fluxo e chamadas de LLM a cada iteração.
    """
    turn_id = _next_turn()
    log_turn_start(turn_id, user_text)

    # injeta turn_id no estado para ficar acessível nos nós (apenas para log)
    state_in: AgentState = {"messages": [HumanMessage(user_text)], "turn_id": turn_id}
    result = graph.invoke(state_in, config=_thread_config())

    assistant_text = str(result["messages"][-1].content)
    log_turn_end(turn_id, assistant_text)
    return assistant_text

# -------------------- CLI opcional --------------------
if __name__ == "__main__":
    graph.get_graph().draw_mermaid_png(output_file_path="graph_flow.png")
    print("Agentic RAG Router com LOG (q para sair)")
    while True:
        user = input("\n> ")
        if user.lower() in {"q", "quit", "exit"}:
            break
        out = agentic_reply(user)
        print(out)
