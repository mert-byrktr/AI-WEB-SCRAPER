from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaEmbeddings, OllamaLLM


class ParseWithOllama:
    def __init__(self, model_name: str = "gemma3", embeddings_name: str = "nomic-embed-text"):
        self.template = (
            "You are tasked with extracting specific information from the following text content: {dom_content}. "
            "Please follow these instructions carefully: \n\n"
            "1. **Extract Information:** Only extract the information that directly matches the provided description: {parse_description}. "
            "2. **No Extra Content:** Do not include any additional text, comments, or explanations in your response. "
            "3. **Empty Response:** If no information matches the description, return an empty string ('')."
            "4. **Direct Data Only:** Your output should contain only the data that is explicitly requested, with no other text."
        )
        self.model, self.local_embeddings = self.load_model_and_embeddings(model_name, embeddings_name)

    def load_model_and_embeddings(self, model_name: str, embeddings_name: str):

        try:
            model = OllamaLLM(model=model_name)
            local_embeddings = OllamaEmbeddings(model=embeddings_name)
            return model, local_embeddings
        except Exception as e:
            print(f"Failed to load model or embeddings: {e}")
            raise

    def format_docs(self, docs: list[Document]):
        return "\n\n".join(doc.page_content for doc in docs)
    
    def parse_with_ollama(self, dom_chunks: list[str], parse_description: str):
        prompt = ChatPromptTemplate.from_template(self.template)
        
        vectorstore = Chroma.from_texts(
            texts=dom_chunks, embedding=self.local_embeddings
        )
        docs = vectorstore.similarity_search(parse_description)
        
        formatted_content = self.format_docs(docs)
        chain = prompt | self.model | StrOutputParser()
        
        response = chain.invoke({
            "dom_content": formatted_content,
            "parse_description": parse_description
        })
        return response
