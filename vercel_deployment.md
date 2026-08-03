# 🚀 Deploying Pakistan Penal Code RAG Chatbot to Vercel

This guide explains whether your GitHub repository should be **Private** or **Public**, and provides step-by-step instructions to deploy your Flask RAG application on Vercel.

---

## 🔒 Private vs. Public GitHub Repository

| Feature | Private Repository (RECOMMENDED 🔒) | Public Repository (🌍) |
| :--- | :--- | :--- |
| **Security** | Protects your code, `.env` file, and internal vector data. | Anyone can view your code. |
| **Vercel Support** | **100% Free & Fully Supported** by Vercel. | Fully Supported by Vercel. |
| **API Key Safety** | Low risk of accidentally committing `.env` API keys. | High risk if `.gitignore` is missing. |

👉 **Recommendation:** Use a **PRIVATE GitHub Repository**. Vercel can seamlessly import and deploy private repositories for free!

---

## 🛠️ Step-by-Step Vercel Deployment Instructions

### Step 1: Push Project to GitHub

Run the following terminal commands in your project folder (`c:\Users\Sheraz\Desktop\RAG`):

```bash
git init
git add .
git commit -m "Deploy PPC Legal RAG Chatbot"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/ppc-rag-chatbot.git
git push -u origin main
```

*(Ensure `.gitignore` is present so `.env` is **not** pushed to GitHub).*

---

### Step 2: Import Project into Vercel

1. Log in to [Vercel](https://vercel.com).
2. Click **"Add New..."** ➔ **"Project"**.
3. Connect your GitHub account and select your **Private** repository (`ppc-rag-chatbot`).
4. In the **Environment Variables** section on Vercel, add:
   - **Key:** `OPENAI_API_KEY`
   - **Value:** `your-openai-api-key-here`
5. Click **"Deploy"**!

---

### Step 3: Access your Live App!

Vercel will build the Python environment and give you a live URL:  
👉 **`https://your-project-name.vercel.app`**
