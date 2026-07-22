import os
import argparse
from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# Ensure you have set OPENAI_API_KEY in your environment variables
# os.environ["OPENAI_API_KEY"] = "your-openai-api-key"

class RAGChatbot:
    def __init__(self, docs_dir="./documents", persist_dir="./chroma_db"):
        self.docs_dir = docs_dir
        self.persist_dir = persist_dir
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        self.chat_history = []
        
    def ingest_documents(self):
        print(f"Loading documents from {self.docs_dir}...")
        if not os.path.exists(self.docs_dir):
            os.makedirs(self.docs_dir)
            print(f"Directory created. Please add PDF or TXT files to '{self.docs_dir}' and run again.")
            return

        loader = DirectoryLoader(self.docs_dir, glob="**/*.*", show_progress=True)
        docs = loader.load()
        
        if not docs:
            print(f"No documents found in {self.docs_dir}.")
            return
            
        print(f"Loaded {len(docs)} documents. Splitting into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200
        )
        splits = text_splitter.split_documents(docs)
        print(f"Created {len(splits)} chunks. Generating embeddings & storing in ChromaDB...")
        
        vectorstore = Chroma.from_documents(
            documents=splits, 
            embedding=self.embeddings, 
            persist_directory=self.persist_dir
        )
        print("Vector database built successfully!")

    def start_chat(self):
        if not os.path.exists(self.persist_dir):
            print("ChromaDB not found. Please run with '--ingest' first to process documents.")
            return

        print("Loading vector database...")
        vectorstore = Chroma(persist_directory=self.persist_dir, embedding_function=self.embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

        # Contextualize question prompt
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        
        history_aware_retriever = create_history_aware_retriever(
            self.llm, retriever, contextualize_q_prompt
        )

        # Answer question prompt
        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer the question. "
            "If you don't know the answer, say that you don't know. "
            "Use three sentences maximum and keep the answer concise.\n\n"
            "{context}"
        )
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        
        question_answer_chain = create_stuff_documents_chain(self.llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        print("\n" + "="*50)
        print("🤖 RAG Chatbot is online! Type 'exit' or 'quit' to stop.")
        print("="*50 + "\n")
        
        while True:
            user_input = input("You: ")
            if user_input.lower() in ['exit', 'quit']:
                break
                
            response = rag_chain.invoke({
                "input": user_input,
                "chat_history": self.chat_history
            })
            
            print(f"\nBot: {response['answer']}\n")
            
            # Update history
            self.chat_history.append(HumanMessage(content=user_input))
            self.chat_history.append(AIMessage(content=response['answer']))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG Chatbot using LangChain and ChromaDB")
    parser.add_argument("--ingest", action="store_true", help="Ingest documents and build vector DB")
    parser.add_argument("--chat", action="store_true", help="Start the interactive chat interface")
    args = parser.parse_args()

    chatbot = RAGChatbot()

    if args.ingest:
        chatbot.ingest_documents()
    elif args.chat:
        chatbot.start_chat()
    else:
        print("Please specify --ingest to load documents or --chat to start the bot.")
