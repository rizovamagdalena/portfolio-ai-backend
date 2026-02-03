from openai import OpenAI
import os


class LLM:
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("Set your OPENAI_API_KEY in .env!")
        self.client = OpenAI(api_key=self.api_key)

    def ask_with_context(self, query: str, context_chunks: list, metadatas: list = None, model="gpt-4o-mini",
                         max_tokens=800):

        if not context_chunks:
            return "I don't have information about that in my project database."

        # Build context with metadata
        if metadatas and len(metadatas) == len(context_chunks):
            context_parts = []
            for i, (chunk, meta) in enumerate(zip(context_chunks, metadatas)):
                context_parts.append(
                    f"[Document {i + 1}]\n"
                    f"Project: {meta['project_name']}\n"
                    f"Type: {meta['chunk_type']}\n"
                    f"Content: {chunk}"
                )
            context_text = "\n\n---\n\n".join(context_parts)
        else:
            # Fallback without metadata
            context_text = "\n---\n".join([f"Document {i + 1}:\n{chunk}" for i, chunk in enumerate(context_chunks)])

        prompt = f"""You are a knowledgeable assistant helping answer questions about a software engineer's project portfolio.

                    CRITICAL INSTRUCTIONS:
                    - Answer using ONLY the information in the context below
                    - ALWAYS mention specific project names when answering (e.g., "MealMakeApp", "StyleCast")
                    - For "which projects" questions, list ALL relevant project names clearly
                    - Be specific and structured in your responses
                    - If the context doesn't fully answer the question, say what you know and what's missing
                    - DO NOT invent or infer information not in the context
                    
                    CONTEXT:
                    {context_text}
                    
                    QUESTION: {query}
                    
                    ANSWER (remember to use specific project names):"""

        response = self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.2
        )
        return response.choices[0].message.content.strip()