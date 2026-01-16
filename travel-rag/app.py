import os
from dotenv import load_dotenv

from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

from models import TravelPlan

# Load environment variables
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(BASE_DIR, ".env")

load_dotenv(dotenv_path=ENV_PATH)


# ---------- Document Loading ----------

DATA_DIR = os.path.join(BASE_DIR, "data")

loader = DirectoryLoader(DATA_DIR, glob="*.txt")
docs = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
documents = text_splitter.split_documents(docs)

# ---------- Local Embeddings ----------

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)


vectorstore = FAISS.from_documents(documents, embeddings)

retriever = vectorstore.as_retriever()

# ---------- OpenRouter LLM ----------

llm = ChatOpenAI(
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    model="mistralai/devstral-2512:free",
    openai_api_base=os.getenv("OPENAI_BASE_URL"),
    temperature=0
)

# ---------- Pydantic Output Parser ----------

parser = PydanticOutputParser(pydantic_object=TravelPlan)

prompt = PromptTemplate(
    template="""
You are a travel planning assistant.

Use the following context to create a travel plan.

Context:
{context}

User Query: {input}

IMPORTANT:
- Respond ONLY with valid JSON
- Do NOT include markdown
- Do NOT include code blocks
- Do NOT include explanations

Return answer strictly in this format:
{format_instructions}
""",
    input_variables=["context", "input"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

# ---------- LCEL Chain ----------

chain = (
    {
        "context": retriever,
        "input": RunnablePassthrough()
    }
    | prompt
    | llm
)

def ask_travel_assistant(query: str):
    result = chain.invoke(query)

    raw = result.content.strip()

    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    try:
        validated_result = parser.parse(raw.strip())
        return validated_result
    except Exception as e:
        print("Validation failed:", e)
        print("Raw Output:", raw)


if __name__ == "__main__":
    
    print("Enter your travel query: ")


    while True:
        user_query = input()
        response = ask_travel_assistant(user_query)

        print("Travel Plan Response:")
        print(response)
