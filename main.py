import os
import sys
from dotenv import load_dotenv

load_dotenv()

def main():
    print("=" * 65)
    print("      PAKISTAN PENAL CODE & CRIMINAL LAW RAG CHATBOT       ")
    print("     Trained on Pakistan Penal Code (1860) & Law Books     ")
    print("=" * 65)

    from vector_store import VectorStore
    store = VectorStore()

    if not store.is_indexed():
        print("\n[*] Local vector store index not found.")
        print("[*] Starting automatic indexing of 'pakistan penal code.pdf'...")
        from indexer import build_index
        build_index()
        print("[OK] Indexing complete!\n")

    from rag_chain import RAGChain
    print("\n[*] Initializing Legal RAG Chain (Model: gpt-4o-mini)...")
    rag = RAGChain(model_name="gpt-4o-mini", top_k=5)
    print("[OK] Legal Chatbot ready! Type 'exit' or 'quit' to stop.\n")
    print("-" * 65)

    while True:
        try:
            user_input = input("\nYou: ").strip()
            if not user_input:
                continue

            if user_input.lower() in ["exit", "quit", "q"]:
                print("\nGoodbye!")
                break

            if user_input.lower() in ["clear", "reset"]:
                rag.clear_history()
                print("[OK] Chat history cleared.")
                continue

            print("\nAssistant: ", end="", flush=True)

            response_stream, sources = rag.ask(user_input, stream=True)
            full_response = ""

            for chunk in response_stream:
                if chunk.choices[0].delta.content:
                    text_delta = chunk.choices[0].delta.content
                    print(text_delta, end="", flush=True)
                    full_response += text_delta

            rag.conversation_history.append({"role": "user", "content": user_input})
            rag.conversation_history.append({"role": "assistant", "content": full_response})

            print("\n\n--- Legal Source References ---")
            unique_sources = set()
            for src in sources:
                doc_name = src.get('document', 'Legal Book')
                page_num = src['page']
                key = f"{doc_name}-P{page_num}"
                if key not in unique_sources:
                    unique_sources.add(key)
                    snippet = src['text'].replace('\n', ' ')[:110]
                    print(f"  • {doc_name} | Page {page_num} (Score: {src['similarity_score']:.2f}): \"{snippet}...\"")

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\n[!] Error: {e}")

if __name__ == "__main__":
    main()
