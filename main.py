import os

# Tavily API 키 직접 하드코딩 (**경고:** 보안 취약점! 실제 서비스에서는 환경변수 등을 사용하세요.)
os.environ["TAVILY_API_KEY"] = "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"

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain import hub
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser  # PydanticOutputParser 추가
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.schema import Document
from pydantic import BaseModel, Field

# 1. 데이터 로딩 및 인덱싱
urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

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

# 2. LLM 및 프롬프트 설정
llm = OllamaLLM(model="llama2", temperature=0)

class GradeDocuments(BaseModel):
    binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")

grade_parser = PydanticOutputParser(pydantic_object=GradeDocuments)  # 파서 생성

system_grade = """You are a grader assessing relevance of a retrieved document to a user question. \n 
    If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_grade),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
)

prompt_rag = hub.pull("rlm/rag-prompt")
rag_chain = prompt_rag | llm | StrOutputParser()

system_rewrite = """You are a question re-writer that converts an input question to a better version that is optimized \n 
     for web search. Look at the input and try to reason about the underlying semantic intent / meaning."""
re_write_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_rewrite),
        (
            "human",
            "Here is the initial question: \n\n {question} \n Formulate an improved question.",
        ),
    ]
)
question_rewriter = re_write_prompt | llm | StrOutputParser()

web_search_tool = TavilySearchResults(k=3)

# 3. 함수 정의

def retrieve(question):
    print("---RETRIEVE---")

    # Encode to UTF-16 and then decode back to UTF-8 to handle surrogates
    question_encoded = question.encode("utf-16", "surrogatepass").decode("utf-16") # add encode and decode
    documents = retriever.invoke(question_encoded) # pass encoded question

    return documents, question


def generate(documents, question):
    print("---GENERATE---")
    generation = rag_chain.invoke({"context": documents, "question": question})
    return generation

def grade_documents(documents, question):
    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    filtered_docs = []
    web_search = "No"
    for d in documents:
        llm_output = llm(grade_prompt.format(question=question, document=d.page_content))  # LLM 직접 호출
        try:
            score = grade_parser.parse(llm_output)  # 파싱
            grade = score.binary_score
        except Exception as e:
            print(f"LLM 출력 파싱 중 오류 발생: {e}")
            print(f"LLM 출력: {llm_output}")
            grade = "no"  # 오류 처리
        if grade == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            web_search = "Yes"
            continue
    return filtered_docs, question, web_search

def transform_query(question, documents):
    print("---TRANSFORM QUERY---")
    better_question = question_rewriter.invoke({"question": question})
    return better_question, documents

def web_search(question, documents):
    print("---WEB SEARCH---")
    docs = web_search_tool.invoke({"query": question})
    web_results = "\n".join([d for d in docs])  # Corrected line
    web_results = Document(page_content=web_results)
    documents.append(web_results)
    return documents, question

def decide_to_generate(web_search):
    print("---ASSESS GRADED DOCUMENTS---")
    if web_search == "Yes":
        print("---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---")
        return "transform_query"
    else:
        print("---DECISION: GENERATE---")
        return "generate"

# 4. 실행
def rag_pipeline(question):
    documents, question = retrieve(question)
    filtered_docs, question, web_search_result = grade_documents(documents, question) # rename web_search to web_search_result
    decision = decide_to_generate(web_search_result)

    if decision == "transform_query":
        better_question, documents = transform_query(question, filtered_docs)
        documents, question = web_search(better_question, documents) # Use better_question as argument
        generation = generate(documents, question)
    else:
        generation = generate(filtered_docs, question)
    return generation



if __name__ == "__main__":
    user_question = input("질문을 입력하세요: ")
    answer = rag_pipeline(user_question)
    print("답변:", answer)
