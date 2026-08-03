import os
from flask import Flask, render_template_string, request, jsonify
from rag_chain import RAGChain
from vector_store import VectorStore
from indexer import build_index

app = Flask(__name__)

rag_chain = None

def get_rag():
    global rag_chain
    if rag_chain is None:
        store = VectorStore()
        if not store.is_indexed():
            print("[*] Vector store missing. Building index...")
            try:
                from indexer import build_index
                build_index()
            except Exception as e:
                print(f"[!] Unable to build index: {e}")
        rag_chain = RAGChain(model_name="gpt-4o-mini", top_k=5)
    return rag_chain

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Pakistan Penal Code (PPC) RAG Chatbot</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <!-- Marked.js CDN for Markdown Parsing -->
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <style>
        :root {
            --bg-primary: #0f172a;
            --bg-secondary: #1e293b;
            --card-bg: #1e293b;
            --text-primary: #f8fafc;
            --text-muted: #94a3b8;
            --accent-color: #38bdf8;
            --accent-hover: #0284c7;
            --user-msg-bg: #0369a1;
            --bot-msg-bg: #1e293b;
            --border-color: #334155;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Inter', sans-serif;
        }

        body {
            background-color: var(--bg-primary);
            color: var(--text-primary);
            height: 100vh;
            display: flex;
            flex-direction: column;
        }

        header {
            background-color: var(--bg-secondary);
            padding: 1rem 2rem;
            border-bottom: 1px solid var(--border-color);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .header-title h1 {
            font-size: 1.25rem;
            font-weight: 700;
            color: var(--accent-color);
        }

        .header-title p {
            font-size: 0.85rem;
            color: var(--text-muted);
        }

        .badge {
            background-color: rgba(56, 189, 248, 0.1);
            color: var(--accent-color);
            border: 1px solid rgba(56, 189, 248, 0.3);
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 600;
        }

        .chat-container {
            flex: 1;
            overflow-y: auto;
            padding: 2rem;
            display: flex;
            flex-direction: column;
            gap: 1.5rem;
            max-width: 900px;
            width: 100%;
            margin: 0 auto;
        }

        .message {
            display: flex;
            flex-direction: column;
            max-width: 88%;
            animation: fadeIn 0.3s ease-in-out;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(8px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .message.user {
            align-self: flex-end;
        }

        .message.assistant {
            align-self: flex-start;
        }

        .message-bubble {
            padding: 1.2rem 1.5rem;
            border-radius: 1rem;
            line-height: 1.7;
            font-size: 0.95rem;
        }

        .message.user .message-bubble {
            background-color: var(--user-msg-bg);
            color: #ffffff;
            border-bottom-right-radius: 0.2rem;
            white-space: pre-wrap;
        }

        .message.assistant .message-bubble {
            background-color: var(--bot-msg-bg);
            color: var(--text-primary);
            border-bottom-left-radius: 0.2rem;
            border: 1px solid var(--border-color);
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
        }

        /* Typography Styling inside Assistant Bubble */
        .message.assistant .message-bubble h1,
        .message.assistant .message-bubble h2,
        .message.assistant .message-bubble h3,
        .message.assistant .message-bubble h4 {
            color: var(--accent-color);
            margin-top: 1rem;
            margin-bottom: 0.5rem;
            font-weight: 600;
            line-height: 1.3;
        }

        .message.assistant .message-bubble h1 { font-size: 1.3rem; border-bottom: 1px solid var(--border-color); padding-bottom: 0.3rem; }
        .message.assistant .message-bubble h2 { font-size: 1.15rem; }
        .message.assistant .message-bubble h3 { font-size: 1.05rem; }
        .message.assistant .message-bubble h4 { font-size: 0.98rem; }

        .message.assistant .message-bubble p {
            margin-bottom: 0.75rem;
        }

        .message.assistant .message-bubble ul,
        .message.assistant .message-bubble ol {
            margin-left: 1.5rem;
            margin-bottom: 0.85rem;
        }

        .message.assistant .message-bubble li {
            margin-bottom: 0.35rem;
        }

        .message.assistant .message-bubble strong {
            color: #f1f5f9;
            font-weight: 600;
        }

        .message.assistant .message-bubble blockquote {
            border-left: 3px solid var(--accent-color);
            padding-left: 1rem;
            margin: 0.75rem 0;
            color: var(--text-muted);
            font-style: italic;
        }

        .sources-container {
            margin-top: 0.75rem;
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
        }

        .source-pill {
            background-color: rgba(56, 189, 248, 0.08);
            border: 1px solid rgba(56, 189, 248, 0.25);
            color: var(--accent-color);
            padding: 0.25rem 0.7rem;
            border-radius: 0.375rem;
            font-size: 0.75rem;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.2s;
        }

        .source-pill:hover {
            background-color: rgba(56, 189, 248, 0.2);
            border-color: var(--accent-color);
        }

        .input-container {
            background-color: var(--bg-secondary);
            padding: 1.25rem 2rem;
            border-top: 1px solid var(--border-color);
        }

        .input-form {
            max-width: 900px;
            margin: 0 auto;
            display: flex;
            gap: 1rem;
        }

        textarea {
            flex: 1;
            background-color: var(--bg-primary);
            border: 1px solid var(--border-color);
            color: var(--text-primary);
            padding: 0.85rem 1.25rem;
            border-radius: 0.75rem;
            font-size: 0.95rem;
            outline: none;
            resize: none;
            height: 52px;
            transition: border-color 0.2s;
        }

        textarea:focus {
            border-color: var(--accent-color);
        }

        button {
            background-color: var(--accent-color);
            color: #0f172a;
            border: none;
            padding: 0 1.5rem;
            border-radius: 0.75rem;
            font-weight: 600;
            font-size: 0.95rem;
            cursor: pointer;
            transition: background-color 0.2s;
        }

        button:hover {
            background-color: var(--accent-hover);
        }

        .typing-indicator {
            display: flex;
            gap: 0.4rem;
            padding: 0.5rem;
        }

        .dot {
            width: 8px;
            height: 8px;
            background-color: var(--text-muted);
            border-radius: 50%;
            animation: bounce 1.4s infinite ease-in-out both;
        }

        .dot:nth-child(1) { animation-delay: -0.32s; }
        .dot:nth-child(2) { animation-delay: -0.16s; }

        @keyframes bounce {
            0%, 80%, 100% { transform: scale(0); }
            40% { transform: scale(1); }
        }
    </style>
</head>
<body>
    <header>
        <div class="header-title">
            <h1>Pakistan Penal Code (PPC) Legal RAG Chatbot</h1>
            <p>Trained on The Pakistan Penal Code (PPC 1860) & Criminal Law</p>
        </div>
        <span class="badge">PPC Active • GPT-4o-mini</span>
    </header>

    <div class="chat-container" id="chatContainer">
        <div class="message assistant">
            <div class="message-bubble">Assalam-o-Alaikum! I am your AI Legal Assistant trained on the <b>Pakistan Penal Code (PPC 1860)</b>.<br><br>Ask me any question or describe a situation:
            <ul>
                <li><b>Which PPC sections apply</b> to a specific incident or crime?</li>
                <li><b>What are your legal rights</b> (Private defense, arrest rules, bail, FIR rights)?</li>
            </ul>
            </div>
        </div>
    </div>

    <div class="input-container">
        <form class="input-form" id="chatForm">
            <textarea id="userInput" placeholder="Ask about applicable PPC sections or your legal rights in a situation..." required></textarea>
            <button type="submit" id="sendBtn">Send</button>
        </form>
    </div>

    <script>
        const chatContainer = document.getElementById('chatContainer');
        const chatForm = document.getElementById('chatForm');
        const userInput = document.getElementById('userInput');

        // Configure Marked options
        marked.setOptions({
            breaks: true,
            gfm: true
        });

        userInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                chatForm.dispatchEvent(new Event('submit'));
            }
        });

        chatForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const text = userInput.value.trim();
            if (!text) return;

            appendMessage('user', text);
            userInput.value = '';

            const typingDiv = document.createElement('div');
            typingDiv.className = 'message assistant';
            typingDiv.id = 'typingIndicator';
            typingDiv.innerHTML = `<div class="message-bubble"><div class="typing-indicator"><div class="dot"></div><div class="dot"></div><div class="dot"></div></div></div>`;
            chatContainer.appendChild(typingDiv);
            chatContainer.scrollTop = chatContainer.scrollHeight;

            try {
                const response = await fetch('/api/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ question: text })
                });
                const data = await response.json();
                
                document.getElementById('typingIndicator')?.remove();

                if (data.answer) {
                    appendMessage('assistant', data.answer, data.sources);
                } else {
                    appendMessage('assistant', 'Error: Unable to generate response.');
                }
            } catch (err) {
                document.getElementById('typingIndicator')?.remove();
                appendMessage('assistant', 'Error connecting to server.');
            }
        });

        function appendMessage(role, content, sources = []) {
            const msgDiv = document.createElement('div');
            msgDiv.className = `message ${role}`;
            
            // Format Markdown for Assistant, plain text for User
            let formattedContent = (role === 'assistant') ? marked.parse(content) : escapeHtml(content);
            let html = `<div class="message-bubble">${formattedContent}</div>`;
            
            if (sources && sources.length > 0) {
                html += `<div class="sources-container">`;
                const seenSources = new Set();
                sources.forEach(src => {
                    const doc = src.document || 'PPC';
                    const key = `${doc}-P${src.page}`;
                    if (!seenSources.has(key)) {
                        seenSources.add(key);
                        html += `<span class="source-pill" title="${escapeHtml(src.text)}">📖 ${doc} - Page ${src.page}</span>`;
                    }
                });
                html += `</div>`;
            }

            msgDiv.innerHTML = html;
            chatContainer.appendChild(msgDiv);
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }

        function escapeHtml(text) {
            return text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
        }
    </script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route("/api/chat", methods=["POST"])
def chat_api():
    try:
        data = request.get_json() or {}
        question = data.get("question", "")
        if not question:
            return jsonify({"error": "Empty question"}), 400

        rag = get_rag()
        result = rag.ask(question, stream=False)
        return jsonify(result)
    except Exception as e:
        print(f"[!] Error handling chat API: {e}")
        return jsonify({"error": str(e), "answer": f"Server Error: {str(e)}"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n[OK] Starting Pakistan Penal Code Web Chat Server at http://127.0.0.1:{port}\n")
    app.run(host="0.0.0.0", port=port, debug=False)
