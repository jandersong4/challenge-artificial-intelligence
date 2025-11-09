from flask import Flask, render_template, jsonify, request #Renderizar html e fazer requisições
from src.helper import downloald_hugging_face_embeddings #cria objeto de embedding do hugging face
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os

PINECONE_INDEX_NAME = "chatbot" #nome do índice no pinecone
GPT_MODEL = "gpt-4.1-nano" #modelo GPT usado

app = Flask(__name__) #Instância do Flask

load_dotenv() #carrega variáveis de ambiente do arquivo .env

PINECODE_API_KEY = os.getenv("PINECONE_API_KEY") #carrega chave da pinecone do .env
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") #carrega chave da openai do .env

os.environ["PINECONE_API_KEY"] = PINECODE_API_KEY #define variável de ambiente para pinecone
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY #define variável de ambiente para openai

embeddings = downloald_hugging_face_embeddings() #cria objeto de embedding do hugging face. Por padrão no metodo já foi definido um modelo de embedding
#Vai ser usado para converter textos em vetores na busca do pinecone

docsearch = PineconeVectorStore.from_existing_index( #Conecta a um índice existente no pinecone e configuraos vetorizer(os embeddings) para consulta
    index_name=PINECONE_INDEX_NAME,
    embedding=embeddings,
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3}) #Cria um objeto retriever para buscar documentos similares no índice pinecone. Retorna os 3 documentos mais relevantes

chatModel = ChatOpenAI(model=GPT_MODEL) #Cria um objeto de modelo de chat usando OpenAI

prompt = ChatPromptTemplate.from_messages([ #Define o prompt de system e user
    ("system", system_promp),
    ("human", "{input}"),
])

question_answer_chain = create_stuff_documents_chain(chatModel,prompt) #Substitui a variável no prompt system pelo conteúdo obtido no RAG
rag_chain = create_retrieval_chain(retriever,question_answer_chain) #Monta a cadeia de RAG completa: ele usa o retriever para buscar docs e passa esses
#docs para a question_answer_chain

@app.route("/") #rota que renderiza o template chat.html
def index():
    return render_template("chat.html")

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"] #recebe mensagem do usuário por um forms
    input = msg
    docs = retriever.get_relevant_documents(msg) #usa o retriever para buscar documentos relevantes no pinecone
    context = "\n\n".join([d.page_content for d in docs]) #concatena o conteúdo dos documentos retornados para usar como contexto no prompt
    response = rag_chain.invoke({"input": msg}) #invoca a cadeia de RAG com a mensagem do usuário
    return str(response["answer"]) #retorna a resposta do modelo como string

if __name__ == '__main__': #executa o app flask
    app.run(host="0.0.0.0", port=8080, debug=True)