import os
import logging
from flask import Flask, request, jsonify
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain import hub
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.schema import Document
from pydantic import BaseModel, Field

# 로그 설정
logging.basicConfig(
    filename="flask.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Flask 기본 로그도 파일에 기록되도록 설정
log = logging.getLogger("werkzeug")
log.setLevel(logging.INFO)
handler = logging.FileHandler("flask.log")
log.addHandler(handler)

logger.info("Flask RAG 서버 시작!")

# 환경 변수 설정
os.environ["TAVILY_API_KEY"] = "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"

# 데이터 로딩 및 인덱싱
urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

logger.info("웹 문서 로딩 시작...")
docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=250, chunk_overlap=0)
doc_splits = text_splitter.split_documents(docs_list)

vectorstore = Chroma.from_documents(
    documents=doc_splits,
    collection_name="rag-chroma",
    embedding=OllamaEmbeddings(model="llama2"),
)
retriever = vectorstore.as_retriever()
logger.info("벡터스토어 생성 완료.")

# LLM 및 프롬프트 설정
llm = OllamaLLM(model="llama2", temperature=0)

class GradeDocuments(BaseModel):
    binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")

grade_parser = PydanticOutputParser(pydantic_object=GradeDocuments)

prompt_rag = hub.pull("rlm/rag-prompt")
rag_chain = prompt_rag | llm | StrOutputParser()

question_rewriter = ChatPromptTemplate.from_messages([
    ("system", "You are a question re-writer that optimizes queries for web search."),
    ("human", "Here is the initial question: {question} \n Formulate an improved question."),
]) | llm | StrOutputParser()

web_search_tool = TavilySearchResults(k=3)

# 📌 **grade_prompt 수정 (LLM이 JSON 형식으로 응답하도록 변경)**
system_grade = """
You are a grader assessing relevance of a retrieved document to a user question.
Your output must be a JSON object in the following format:
{
  "binary_score": "yes" or "no"
}
If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant with "yes".
Otherwise, return "no".
Your response must be valid JSON, without extra explanation.
"""

grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_grade),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
)

# 함수 정의
def retrieve(question):
    logger.info(f"질문 수신: {question}")
    question_encoded = question.encode("utf-16", "surrogatepass").decode("utf-16")
    documents = retriever.invoke(question_encoded)
    logger.info(f"검색된 문서 개수: {len(documents)}")
    return documents, question

def generate(documents, question):
    logger.info("응답 생성 시작")
    generation = rag_chain.invoke({"context": documents, "question": question})
    logger.info(f"생성된 응답: {generation}")
    return generation

def grade_documents(documents, question):
    logger.info("문서 유효성 평가 시작")
    filtered_docs = []
    web_search = "No"
    for d in documents:
        try:
            llm_output = llm.invoke(grade_prompt.format(question=question, document=d.page_content))
            score = grade_parser.parse(llm_output)  # JSON으로 변환된 결과 처리
            grade = score.binary_score
        except Exception as e:
            logger.error(f"LLM 출력 파싱 중 오류 발생: {e}")
            logger.error(f"LLM 출력: {llm_output if 'llm_output' in locals() else 'N/A'}")
            grade = "no"
        if grade == "yes":
            logger.info("문서 관련성 있음")
            filtered_docs.append(d)
        else:
            logger.info("문서 관련성 없음, 웹 검색 진행")
            web_search = "Yes"
    return filtered_docs, question, web_search

def transform_query(question, documents):
    logger.info("질문 개선 진행")
    better_question = question_rewriter.invoke({"question": question})
    logger.info(f"개선된 질문: {better_question}")
    return better_question, documents

# 📌 **웹 검색 오류 해결 (dict → str 변환)**
def web_search(question, documents):
    logger.info("웹 검색 시작")
    docs = web_search_tool.invoke({"query": question})
    
    # 검색 결과를 문자열로 변환
    web_results = "\n".join([d["content"] if isinstance(d, dict) else str(d) for d in docs])
    
    web_results = Document(page_content=web_results)
    documents.append(web_results)
    logger.info("웹 검색 완료, 문서 추가")
    return documents, question

def decide_to_generate(web_search):
    if web_search == "Yes":
        logger.info("문서가 적절하지 않아 질문 재작성 진행")
        return "transform_query"
    else:
        logger.info("문서가 충분하므로 바로 응답 생성")
        return "generate"

# 실행 함수
def rag_pipeline(question):
    logger.info("RAG 파이프라인 실행 시작")
    try:
        documents, question = retrieve(question)
        filtered_docs, question, web_search_result = grade_documents(documents, question)
        decision = decide_to_generate(web_search_result)

        if decision == "transform_query":
            better_question, documents = transform_query(question, filtered_docs)
            documents, question = web_search(better_question, documents)
            generation = generate(documents, question)
        else:
            generation = generate(filtered_docs, question)
        
        logger.info("RAG 파이프라인 실행 완료")
        return generation
    except Exception as e:
        logger.error(f"RAG 파이프라인 실행 중 오류 발생: {e}")
        return "오류 발생"

# Flask API 설정
app = Flask(__name__)

@app.route("/ask", methods=["POST"])
def ask_question():
    """POST로 {"question": "..."} 형태의 JSON을 받으면, RAG 파이프라인 응답을 반환"""
    data = request.get_json()
    user_question = data.get("question", "")
    logger.info(f"API 요청 수신: {user_question}")
    answer = rag_pipeline(user_question)
    logger.info(f"API 응답 반환: {answer}")
    return jsonify({"answer": answer})

if __name__ == "__main__":
    logger.info("Flask 서버 시작")
    app.run(host="0.0.0.0", port=5000, debug=True)

