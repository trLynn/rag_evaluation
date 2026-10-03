import os
from typing import List, TypedDict
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langgraph.graph import StateGraph, START, END

# --- 1. LANGSMITH TRACING SETUP ---
def setup_langsmith(project_name: str = "rag_evaluation"):
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    # Set your key via environment variable or fill in default
    if "LANGCHAIN_API_KEY" not in os.environ:
        os.environ["LANGCHAIN_API_KEY"] = "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"
    os.environ["LANGCHAIN_PROJECT"] = project_name

# --- 2. GRAPH STATE DEFINITION ---
class GraphState(TypedDict):
    question: str
    documents: List[str]
    generation: str
    is_relevant: bool

# --- 3. ADAPTIVE RAG GRAPH CLASS ---
class AdaptiveRAGGraphEngine:
    def __init__(
        self,
        persist_dir: str = "vector_db",
        collection_name: str = "knowledge_base",
        embedding_model: str = "nomic-embed-text",
        llm_model: str = "llama3.1"
    ):
        setup_langsmith()
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.embeddings = OllamaEmbeddings(model=embedding_model)
        self.llm_model_name = llm_model
        self.llm = ChatOllama(model=llm_model, temperature=0)
        self.app = self._build_graph()

    def _build_graph(self):
        def retrieve(state: GraphState):
            db = Chroma(
                persist_directory=self.persist_dir,
                collection_name=self.collection_name,
                embedding_function=self.embeddings
            )
            retriever = db.as_retriever(search_kwargs={"k": 3})
            docs = retriever.invoke(state["question"])
            return {"documents": [d.page_content for d in docs]}

        def grade_documents(state: GraphState):
            prompt = (
                f"Is the following retrieved context relevant to the question: '{state['question']}'?\n"
                f"Answer strictly 'yes' or 'no'.\n\nContext: {state['documents']}"
            )
            response = self.llm.invoke(prompt).content.strip().lower()
            return {"is_relevant": "yes" in response}

        def generate(state: GraphState):
            prompt = (
                f"You are a helpful assistant. Answer based strictly on the context.\n"
                f"Context: {state['documents']}\nQuestion: {state['question']}"
            )
            response = self.llm.invoke(prompt)
            return {"generation": response.content}

        def transform_query(state: GraphState):
            prompt = (
                f"Rephrase this search query to be more effective for vector database retrieval: "
                f"{state['question']}"
            )
            response = self.llm.invoke(prompt)
            return {"question": response.content}

        def decide_to_generate(state: GraphState):
            return "generate" if state["is_relevant"] else "transform_query"

        # Construct StateGraph
        workflow = StateGraph(GraphState)
        workflow.add_node("retrieve", retrieve)
        workflow.add_node("grade_documents", grade_documents)
        workflow.add_node("generate", generate)
        workflow.add_node("transform_query", transform_query)

        workflow.add_edge(START, "retrieve")
        workflow.add_edge("retrieve", "grade_documents")
        workflow.add_conditional_edges(
            "grade_documents",
            decide_to_generate,
            {"generate": "generate", "transform_query": "transform_query"}
        )
        workflow.add_edge("transform_query", "retrieve")
        workflow.add_edge("generate", END)

        return workflow.compile()

    def run(self, question: str) -> dict:
        result = self.app.invoke({"question": question})
        return {
            "answer": result.get("generation", ""),
            "model_used": self.llm_model_name,
            "documents": result.get("documents", []),
            "is_relevant": result.get("is_relevant", False),
            "final_query": result.get("question", question)
        }