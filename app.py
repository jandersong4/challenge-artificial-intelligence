from flask import Flask, render_template, request
from dotenv import load_dotenv
import os
import uuid
from src.graphChat import agentic_reply  

app = Flask(__name__)

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")
os.environ["PINECONE_API_KEY"] = os.getenv("PINECONE_API_KEY", "")

@app.route("/")
def index():
    try:
        welcome = agentic_reply()
    except Exception as e:
        print("Erro ao obter mensagem inicial:", e)
        welcome = "Olá! Sou a professora Maísa. Estou aqui para conversar sobre o tema deste chat."

    session_id = str(uuid.uuid4())
    return render_template(
        "chat.html",
        welcome_message=welcome,
        session_id=session_id
    )

@app.route("/get", methods=["POST"])
def chat():
    """
    Rota de interação:
    - Recebe mensagem do usuário.
    - Retorna a resposta do agente.
    """
    msg = request.form.get("msg", "").strip()
    if not msg:
        return "Mensagem vazia.", 400

    try:
        response = agentic_reply(msg)
    except Exception as e:
        print("Erro ao processar mensagem:", e)
        response = "Desculpe, ocorreu um erro ao processar sua mensagem."

    return str(response)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)

