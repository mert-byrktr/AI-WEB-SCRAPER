from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser

template = (
    "You are tasked with extracting specific information from the following text content: {dom_content}. "
    "Please follow these instructions carefully: \n\n"
    "1. **Extract Information:** Only extract the information that directly matches the provided description: {parse_description}. "
    "2. **No Extra Content:** Do not include any additional text, comments, or explanations in your response. "
    "3. **Empty Response:** If no information matches the description, return an empty string ('')."
    "4. **Direct Data Only:** Your output should contain only the data that is explicitly requested, with no other text."
)

local_embeddings = OllamaEmbeddings(model="nomic-embed-text")
model = OllamaLLM(model="gemma3")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def parse_with_ollama(dom_chunks, parse_description):
    prompt = ChatPromptTemplate.from_template(template)
    
    vectorstore = Chroma.from_texts(
        texts=dom_chunks, embedding=local_embeddings
    )
    docs = vectorstore.similarity_search(parse_description)
    
    formatted_content = format_docs(docs)
    chain = prompt | model | StrOutputParser()
    
    response = chain.invoke({
        "dom_content": formatted_content,
        "parse_description": parse_description
    })
    return response
