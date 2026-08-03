import os
from openai import OpenAI
from dotenv import load_dotenv
from vector_store import VectorStore

load_dotenv(override=True)

SYSTEM_PROMPT = """You are an expert AI Legal Advisor specializing in the Pakistan Penal Code (PPC 1860) and Pakistani Criminal Law.

CORE RESPONSIBILITIES:

1. APPLICABLE SECTIONS ADVISOR:
   - When a user describes a specific scenario, incident, or crime (e.g., theft, assault, housebreaking, fraud, criminal threats, land dispute, extortion), identify and explain ALL applicable PPC Sections.
   - For each section, provide:
     • Section Number & Title (e.g., PPC Section 379: Theft, Section 420: Cheating).
     • Practical application to the described facts.
     • Prescribed Punishment (imprisonment term, fine).
     • Legal Status (Cognizable/Non-cognizable, Bailable/Non-bailable).

2. USER LEGAL RIGHTS & GUIDANCE ADVISOR:
   - When a user asks about their rights or needs guidance in a scenario (e.g., self-defense, rights during police arrest, protection against false FIR, victim rights), outline their legal rights clearly.
   - Detail the Right of Private Defense (PPC Sections 96–106) where applicable.
   - Provide clear, actionable steps (e.g., lodging an FIR under Section 154 CrPC, pre-arrest bail under Section 498 CrPC, or consulting an advocate).

STRICT RESPONSIBLE CONDUCT RULES:
- NEVER make speculative, sensitive, or offensive assumptions about a user based on their personal name (e.g., "Sheraz Ahmad"). Names are common identifiers and must NEVER trigger assumptions regarding religious laws, sect, or identity unless the user explicitly asks a specific legal question about a section.
- If a user enters a greeting, personal name, or unclear statement without describing an offense or asking a clear legal question, respond politely and invite them to ask about specific PPC sections, legal rights, or crime scenarios.
- Ground all answers strictly in the factual context of the Pakistan Penal Code without making false inferences.
- Include a standard legal disclaimer at the end of responses.
"""

class RAGChain:
    def __init__(self, model_name="gpt-4o-mini", top_k=5):
        load_dotenv(override=True)
        api_key = (os.getenv("OPENAI_API_KEY") or "").strip(' "\'')
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable missing.")

        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name
        self.top_k = top_k
        self.vector_store = VectorStore()
        self.conversation_history = []

    def get_query_embedding(self, query: str):
        response = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=query
        )
        return response.data[0].embedding

    def retrieve_context(self, query: str, top_k=None):
        if top_k is None:
            top_k = self.top_k

        # If query is very short or just a name/greeting, return minimal top-k
        query_words = query.strip().split()
        if len(query_words) <= 3:
            top_k = 3

        query_emb = self.get_query_embedding(query)
        relevant_chunks = self.vector_store.similarity_search(query_emb, top_k=top_k)
        return relevant_chunks

    def format_context_prompt(self, relevant_chunks):
        formatted_pieces = []
        for i, chunk in enumerate(relevant_chunks, 1):
            doc_name = chunk.get("document", "Legal Document")
            piece = f"--- CONTEXT CHUNK {i} (Source: {doc_name} - Page {chunk['page']}) ---\n{chunk['text']}"
            formatted_pieces.append(piece)
        return "\n\n".join(formatted_pieces)

    def ask(self, query: str, stream=False):
        """
        Processes a user query: retrieves context, formats messages with conversation history,
        and generates response from LLM.
        """
        relevant_chunks = self.retrieve_context(query)
        context_str = self.format_context_prompt(relevant_chunks)

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        for msg in self.conversation_history[-6:]:
            messages.append(msg)

        user_content_with_context = (
            f"Retrieved Legal Context:\n{context_str}\n\n"
            f"User Query / Statement: {query}"
        )
        messages.append({"role": "user", "content": user_content_with_context})

        if stream:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.1,
                stream=True
            )
            return response, relevant_chunks
        else:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.1,
                stream=False
            )
            answer = response.choices[0].message.content

            self.conversation_history.append({"role": "user", "content": query})
            self.conversation_history.append({"role": "assistant", "content": answer})

            return {
                "answer": answer,
                "sources": [
                    {
                        "document": c.get("document", "Legal Document"),
                        "page": c["page"],
                        "text": c["text"],
                        "score": round(c["similarity_score"], 4)
                    }
                    for c in relevant_chunks
                ]
            }

    def clear_history(self):
        self.conversation_history = []

if __name__ == "__main__":
    chain = RAGChain()
    test_query = "Hello, my name is Sheraz Ahmad."
    print(f"Testing Name/Greeting Query: '{test_query}'")
    res = chain.ask(test_query)
    print("\n--- RESPONSE ---")
    print(res["answer"])
