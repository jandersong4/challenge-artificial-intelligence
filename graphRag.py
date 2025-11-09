# --- chat_memory.py (pode estar no mesmo arquivo do Flask, acima das rotas) ---
import threading
from collections.abc import Sequence
from typing import Annotated, TypedDict

from langchain.chat_models import init_chat_model
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph, add_messages
from langgraph.graph.state import RunnableConfig
from dotenv import load_dotenv
import os


load_dotenv() #carrega variáveis de ambiente do arquivo .env

PINECODE_API_KEY = os.getenv("PINECONE_API_KEY") #carrega chave da pinecone do .env
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") #carrega chave da openai do .env

os.environ["PINECONE_API_KEY"] = PINECODE_API_KEY #define variável de ambiente para pinecone
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY #define variável de ambiente para openai

# === LLM ===
# Mantive exatamente sua chamada, sem “evoluções”:
llm = init_chat_model("gpt-4.1-nano")

# === State ===
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

# === Node ===
def call_llm(state: AgentState) -> AgentState:
    llm_result = llm.invoke(state["messages"])
    return {"messages": [llm_result]}

# === Grafo (compilado uma única vez) ===
_builder = StateGraph(AgentState, context_schema=None, input_schema=AgentState, output_schema=AgentState)
_builder.add_node("call_llm", call_llm)
_builder.add_edge(START, "call_llm")
_builder.add_edge("call_llm", END)

_checkpointer = InMemorySaver()
_graph = _builder.compile(checkpointer=_checkpointer)

# === Função pública para usar no POST ===
def chat_with_memory(session_id: str, user_text: str) -> str:
    """
    Recebe um identificador de sessão (session_id) e o texto do usuário,
    invoca o grafo com memória e retorna apenas o texto de resposta.
    """
    # O LangGraph usa "thread_id" para manter o histórico por conversa
    config = RunnableConfig(configurable={"thread_id": session_id})
    human_message = HumanMessage(user_text)
    result = _graph.invoke({"messages": [human_message]}, config=config)

    # O último item em result["messages"] é a resposta do modelo
    last_msg = result["messages"][-1]
    return getattr(last_msg, "content", str(last_msg))

teste = chat_with_memory("test_session", "Hello, world!")  # Teste rápido para garantir que funciona
print(teste)