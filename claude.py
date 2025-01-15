import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders.text import TextLoader

# load api key from environment file
load_dotenv()
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")

class TextKnowledgeBaseApp:
    def __init__(self, anthropic_api_key: str):
        # Initialize the ChatAnthropic model
        self.llm = ChatAnthropic(
            model="claude-3-sonnet-20240229",
            anthropic_api_key=anthropic_api_key,
            max_tokens=1024
        )
        
        # by using init_chat_model, we could also write a model chooser here that initiializes like this: 
        # claude_opus = init_chat_model("claude-3-opus-20240229", model_provider="anthropic", temperature=0)
        # gpt_4o = init_chat_model("gpt-4o", model_provider="openai", temperature=0)
        # langchain itself handels the calls to the specified model
        
        # Initialize local embeddings using sentence-transformers
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2", # this is where we can define different embedding models to be used
            model_kwargs={'device': 'cpu'}
            # model_kwargs={'device': 'cuda'} --> use this line instead if gpu is available and torch was compiled with cuda
        )
        
        # Initialize conversation memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        
        self.vector_store = None
        
    def load_text_files(self, text_directory: str, encoding: str = 'utf-8'):
        """Load and process text documents from a directory"""
        documents = []
        
        # Load text files
        for filename in os.listdir(text_directory):
            if filename.endswith('.txt'):
                text_path = os.path.join(text_directory, filename)
                print(f"Current file: {text_path}")
                try:
                    loader = TextLoader(text_path, encoding=encoding)
                    documents.extend(loader.load())
                except UnicodeDecodeError as e:
                    print(f"Error loading {filename}: {e}. Try specifying a different encoding.")
        
        if not documents:
            raise ValueError("No text documents were successfully loaded.")
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        splits = text_splitter.split_documents(documents)
        
        # Create vector store
        self.vector_store = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings
        )
        
        # Initialize the conversation chain
        self.chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vector_store.as_retriever(),
            memory=self.memory,
            return_source_documents=True
        )
        
    def ask_question(self, question: str) -> str:
        """Ask a question and get a response based on the PDF knowledge base"""
        if not self.vector_store:
            return "Please load PDF documents first using load_pdfs()"
        
        # Get response from the chain
        response = self.chain({"question": question})
        
        return response["answer"]

# Example usage
def main():
    app = TextKnowledgeBaseApp(
        anthropic_api_key=CLAUDE_API_KEY
    )
    
    # Load text files from a directory
    try:
        app.load_text_files("training_data/")
    except ValueError as e:
        print(f"Error: {e}")
        return
    
    # Ask questions
    while True:
        question = input("Ask a question (or type 'quit' to exit): ")
        if question.lower() == 'quit':
            break
            
        response = app.ask_question(question)
        print("\nResponse:", response)
        print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    main()