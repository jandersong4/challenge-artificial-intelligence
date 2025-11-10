from flask import Flask, render_template, jsonify, request #Renderizar html e fazer requisições
from src.helper import downloald_hugging_face_embeddings #cria objeto de embedding do hugging face
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
# from langchain.chains import create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os
import threading
from collections.abc import Sequence
from typing import Annotated, TypedDict
from langchain.chat_models import init_chat_model
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph, add_messages
from langgraph.graph.state import RunnableConfig

PINECONE_INDEX_NAME = "chatbot" 
GPT_MODEL = "gpt-4.1-nano" 

app = Flask(__name__) 

load_dotenv() 

PINECODE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECODE_API_KEY 
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY 
embeddings = downloald_hugging_face_embeddings()

docsearch = PineconeVectorStore.from_existing_index(index_name=PINECONE_INDEX_NAME,embedding=embeddings)
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3}) 

chatModel = ChatOpenAI(model=GPT_MODEL) #Cria um objeto de modelo de chat usando OpenAI

# === State ===
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

# === Node ===
def call_llm(state: AgentState) -> AgentState:
    llm_result = chatModel.invoke(state["messages"])
    return {"messages": [llm_result]}

_builder = StateGraph(AgentState, context_schema=None, input_schema=AgentState, output_schema=AgentState)
_builder.add_node("call_llm", call_llm)
_builder.add_edge(START, "call_llm")
_builder.add_edge("call_llm", END)

_checkpointer = InMemorySaver()
_graph = _builder.compile(checkpointer=_checkpointer)

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

@app.route("/") #rota que renderiza o template chat.html
def index():
    return render_template("chat.html")

@app.route("/get", methods=["GET", "POST"])
def chat():
    # 1) Captura a mensagem
    msg = request.form["msg"]

    # 2) Defina um identificador de sessão para manter o histórico.
    #    Você pode receber do front (ex.: form["session_id"]) ou gerar por IP/user-agent.
    session_id = request.form.get("session_id") or request.remote_addr or "default-session"

    # 3) Chama a função de chat com memória
    answer = chat_with_memory(session_id=session_id, user_text=msg)

    # 4) Retorna texto simples (igual você já faz)
    return str(answer)

if __name__ == '__main__': #executa o app flask
    app.run(host="0.0.0.0", port=8080, debug=True)