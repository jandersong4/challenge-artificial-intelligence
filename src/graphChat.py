import threading
from collections.abc import Sequence
from typing import Annotated, TypedDict, Literal, Optional, List, Dict, Any
import os
from src.prompt import classifier_prompt, general_system_prompt, welcome_message
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph, add_messages
from langgraph.graph.state import RunnableConfig
from langchain_pinecone import PineconeVectorStore
from src.helper import downloald_hugging_face_embeddings
from src.agent_logging import AgentLogger

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")
os.environ["PINECONE_API_KEY"] = os.getenv("PINECONE_API_KEY", "")

GPT_MODEL = "openai:gpt-4.1-mini"
PINECONE_INDEX_NAME = "chatbot"

llm = init_chat_model(GPT_MODEL)

embeddings = downloald_hugging_face_embeddings()
docsearch = PineconeVectorStore.from_existing_index(index_name=PINECONE_INDEX_NAME, embedding=embeddings)
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 2})

logger = AgentLogger()

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

_SESSION_PRIMED = {}
def _session_key() -> str:
    return str(threading.get_ident())

def _state_snapshot_for_log(state: AgentState) -> Dict[str, Any]:
    snap: Dict[str, Any] = {}
    if "needs_search" in state:
        snap["needs_search"] = state["needs_search"]
    if "context_chunks" in state:
        snap["context_chunks_len"] = len(state.get("context_chunks") or [])
    if "messages" in state:
        msgs = state.get("messages") or []
        snap["last_message"] = AgentLogger.safe_serialize_messages(msgs[-1:])
        snap["messages_count"] = len(msgs)
    if "turn_id" in state and state["turn_id"] is not None:
        snap["turn_id"] = state["turn_id"]
    return snap

# -------------------- Nós --------------------
def classify_need_search(state: AgentState) -> AgentState:
    turn_id = state.get("turn_id") or 0
    logger.log_node_enter(turn_id, "classify", _state_snapshot_for_log(state))

    user_utterance = _last_user_text(state["messages"])
    sys = SystemMessage(content=classifier_prompt.format(context=user_utterance))
    msgs = [sys, HumanMessage(content=user_utterance)]
    judge = llm.invoke(msgs)

    logger.log_llm_call(turn_id, "classify", msgs, judge)

    decision_text = (judge.content or "").strip().upper()
    needs = decision_text.startswith("Y")

    logger.log_route_decision(turn_id, needs)

    result = {"needs_search": needs}
    logger.log_node_exit(turn_id, "classify", result)
    return result

def retrieve_docs(state: AgentState) -> AgentState:
    turn_id = state.get("turn_id") or 0
    logger.log_node_enter(turn_id, "retrieve", _state_snapshot_for_log(state))

    user_utterance = _last_user_text(state["messages"])
    docs = retriever.invoke(user_utterance) 
    chunks = [d.page_content for d in docs]
    
    logger.log_retrieve(turn_id, user_utterance, docs)
    logger.log_node_exit(turn_id, "retrieve",{"docs_count": len(docs), "chunks_len": len(chunks)})
    
    return {"context_chunks": chunks}

def answer_with_context(state: AgentState) -> AgentState:
    turn_id = state.get("turn_id") or 0
    logger.log_node_enter(turn_id, "answer_with_context", _state_snapshot_for_log(state))

    user_utterance = _last_user_text(state["messages"])
    context = "\n\n".join(state.get("context_chunks") or [])
    sys = SystemMessage(content=(
        general_system_prompt
        + "\n\nUse APENAS as informações a seguir como contexto quando forem relevantes."
        + " Se o contexto não contiver a resposta, seja honesto."
        + f"\n\n[CONTEXT]\n{context}\n[/CONTEXT]"
    ))
    msgs = [sys] + state.get("messages", [])
    resp = llm.invoke(msgs)

    logger.log_llm_call(turn_id, "answer_with_context", msgs, resp)

    result = {"messages": [resp]}
    logger.log_node_exit(turn_id, "answer_with_context", {"assistant_preview": str(resp.content)[:200]})
    return result

def answer_direct(state: AgentState) -> AgentState:
    turn_id = state.get("turn_id") or 0
    logger.log_node_enter(turn_id, "answer_direct", _state_snapshot_for_log(state))

    msgs = state.get("messages", [])
    resp = llm.invoke(msgs)

    logger.log_llm_call(turn_id, "answer_direct", msgs, resp)
    result = {"messages": [resp]}
    logger.log_node_exit(turn_id, "answer_direct", {"assistant_preview": str(resp.content)[:200]})
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
def agentic_reply(user_text: Optional[str] = None) -> str:
    """
    Regra:
      - Se a sessão acabou de iniciar e ainda não houve input do usuário,
        mostra a mensagem de boas-vindas (NÃO roda o grafo ainda).
      - Na PRIMEIRA mensagem do usuário, enviamos [System, AI(welcome), Human]
        para o grafo, de modo que a saudação faça parte do contexto.
      - Nos turnos seguintes, apenas a HumanMessage é adicionada; o histórico
        completo já está salvo pelo checkpointer do LangGraph.
    """
    session = _session_key()
    primed = _SESSION_PRIMED.get(session, False)

    if not primed and (user_text is None or not user_text.strip()):
        turn_id = logger.next_turn()
        logger.log_turn_start(turn_id, "(inicialização do chat)")
        logger.log_turn_end(turn_id, welcome_message)
        return welcome_message

    if not primed:
        turn_id = logger.next_turn()
        logger.log_turn_start(turn_id, user_text)

        init_messages: Sequence[BaseMessage] = [
            SystemMessage(content=general_system_prompt),
            AIMessage(content=welcome_message),
            HumanMessage(user_text)
        ]

        state_in: AgentState = {"messages": init_messages, "turn_id": turn_id}
        result = graph.invoke(state_in, config=_thread_config())

        assistant_text = str(result["messages"][-1].content)
        logger.log_turn_end(turn_id, assistant_text)

        _SESSION_PRIMED[session] = True
        return assistant_text

    turn_id = logger.next_turn()
    logger.log_turn_start(turn_id, user_text)

    state_in: AgentState = {"messages": [HumanMessage(user_text)], "turn_id": turn_id}
    result = graph.invoke(state_in, config=_thread_config())

    assistant_text = str(result["messages"][-1].content)
    logger.log_turn_end(turn_id, assistant_text)
    return assistant_text


# -------------------- CLI --------------------
if __name__ == "__main__":
    graph.get_graph().draw_mermaid_png(output_file_path="graph_flow.png")
    print("Agentic RAG Router com LOG (q para sair)")

    print(agentic_reply())

    while True:
        user = input("\n> ")
        out = agentic_reply(user)
        print(out)