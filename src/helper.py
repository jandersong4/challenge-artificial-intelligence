from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from typing import List
from langchain_core.documents import Document
from collections import Counter
import fitz

BLACK_FONT = "MyriadPro-Black"
SEMIBOLD_FONT = "MyriadPro-Semibold"

def _dominant_font(spanList):
    """
    Retorna a fonte dominante (mais frequente) dentre os spans com texto.
    """
    fonts = []
    for span in spanList:
        text = span.get("text", "").strip()
        if text:
            fonts.append(span.get("font", ""))
    if not fonts:
        return ""
    return Counter(fonts).most_common(1)[0][0]      

def _iter_lines(doc):
    """
    Itera todas as linhas do PDF na ordem de leitura,
    retornando tuplas (text, font).
    """
    for page in doc:
        data = page.get_text("dict")
        for block in data.get("blocks", []):
            if "lines" not in block:
                continue
            for line in block["lines"]:
                spansList = line.get("spans", [])
                text = ""
                for span in spansList:
                    text += span.get("text", "")
                text = text.strip()
                if not text:
                    continue
                font = _dominant_font(spansList)
                yield text, font

def _is_black(font: str) -> bool:
    return font and BLACK_FONT.lower() in font.lower()

def _is_semibold(font: str) -> bool:
    return font and SEMIBOLD_FONT.lower() in font.lower()

def extract_sections_as_documents(pdf_path: str) -> List[Document]:
    """
    Divide o PDF em seções:
      - Seção começa em linha(s) com fonte MyriadPro-Black (título pode ter várias linhas seguidas).
      - Conteúdo vai até a próxima ocorrência de título (MyriadPro-Black) ou fim.
      - Keywords = subseções (MyriadPro-Semibold) encontradas dentro da seção + o próprio título.
    Retorna uma lista de langchain.schema.Document.
    """
    doc = fitz.open(pdf_path)
    lines = list(_iter_lines(doc))
    i = 0
    number_of_lines = len(lines)
    docs: List[Document] = []

    while i < number_of_lines:
        text, font = lines[i]
 
        if _is_black(font):
            title_parts = [text]
            j = i + 1
            while j < number_of_lines and _is_black(lines[j][1]):
                title_parts.append(lines[j][0])
                j += 1
            title = " ".join(part.strip() for part in title_parts).strip()

            content_lines = []
            keywords = set()
            keywords.add(title)

            k = j
            while k < number_of_lines and not _is_black(lines[k][1]):
                line_text, line_font = lines[k]
                if _is_semibold(line_font):
                    keywords.add(line_text.strip())
                content_lines.append(line_text)
                k += 1

            section_text = title + "\n\n" + "\n".join(content_lines).strip()
            docs.append(
                Document(
                    page_content=section_text,
                    metadata={
                        "source": pdf_path,
                        "title": title,
                        "keywords": sorted(list(keywords)),
                    },
                )
            )
            i = k 
        else:
            i += 1

    doc.close()
    return docs

def downloald_embeddings():
    model_name = "BAAI/bge-small-en-v1.5"
    embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs={"device": "cpu"})
    return embeddings

def downloald_hugging_face_embeddings():
    model_name = "BAAI/bge-small-en-v1.5"
    embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs={"device": "cpu"})
    return embeddings

