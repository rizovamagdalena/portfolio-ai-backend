import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


class LLM:
    PROMPT_TEMPLATE = """You are a knowledgeable assistant helping answer questions about a software engineer's project portfolio.

CRITICAL INSTRUCTIONS:
- Answer using ONLY the information in the context below
- ALWAYS mention specific project names when answering (e.g., "MealMakeApp", "StyleCast")
- For "which projects" questions, list ALL relevant project names clearly
- Be specific and structured in your responses
- If the context doesn't fully answer the question, say what you know and what's missing
- DO NOT invent or infer information not in the context

CONTEXT:
{context}

QUESTION: {question}

ANSWER (remember to use specific project names):"""

    def __init__(self, model="gpt-4o-mini", temperature=0.2, max_tokens=800):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("Set your OPENAI_API_KEY in .env!")

        # Store default parameters
        self.default_model = model
        self.default_temperature = temperature
        self.default_max_tokens = max_tokens

        # Initialize LangChain ChatOpenAI
        self.llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            openai_api_key=self.api_key
        )

        # Create reusable prompt
        self.prompt = ChatPromptTemplate.from_template(self.PROMPT_TEMPLATE)

    def _get_llm(self, model=None, max_tokens=None):
        if model is None and max_tokens is None:
            return self.llm

        return ChatOpenAI(
            model=model or self.default_model,
            temperature=self.default_temperature,
            max_tokens=max_tokens or self.default_max_tokens,
            openai_api_key=self.api_key
        )

    @staticmethod
    def _format_context(context_chunks: list, metadatas: list = None) -> str:
        if metadatas and len(metadatas) == len(context_chunks):
            context_parts = []
            for i, (chunk, meta) in enumerate(zip(context_chunks, metadatas)):
                context_parts.append(
                    f"[Document {i + 1}]\n"
                    f"Project: {meta['project_name']}\n"
                    f"Type: {meta['chunk_type']}\n"
                    f"Content: {chunk}"
                )
            return "\n\n---\n\n".join(context_parts)
        else:
            # Fallback without metadata
            return "\n---\n".join([f"Document {i + 1}:\n{chunk}" for i, chunk in enumerate(context_chunks)])

    def ask_with_context(self, query: str, context_chunks: list, metadatas: list = None,
                         model=None, max_tokens=None):
        if not context_chunks:
            return "I don't have information about that in my project database."

        # Format context
        context_text = self._format_context(context_chunks, metadatas)

        # Get appropriate LLM
        llm = self._get_llm(model, max_tokens)

        # Create and invoke chain
        chain = (
                {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
                | self.prompt
                | llm
                | StrOutputParser()
        )

        response = chain.invoke({"context": context_text, "question": query})
        return response.strip()

    def create_rag_chain(self, retriever):

        def format_docs(docs):
            context_parts = []
            for i, doc in enumerate(docs):
                meta = doc.metadata
                context_parts.append(
                    f"[Document {i + 1}]\n"
                    f"Project: {meta.get('project_name', 'Unknown')}\n"
                    f"Type: {meta.get('chunk_type', 'General')}\n"
                    f"Content: {doc.page_content}"
                )
            return "\n\n---\n\n".join(context_parts)

        # Create RAG chain
        rag_chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | self.prompt
                | self.llm
                | StrOutputParser()
        )

        return rag_chain