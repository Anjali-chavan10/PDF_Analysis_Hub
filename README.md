

# 📊 PDF Analysis Hub

A powerful Streamlit-based web application that enables intelligent analysis and question-answering on PDF documents — including both normal and scanned PDFs — using **Google Gemini API**, **LangChain**, **FAISS**, and **OCR (Optical Character Recognition)** techniques.

---

## 🚀 Features

- 🔐 **Secure Gemini API Key Integration**
- 📄 **Normal PDF Analysis** (embedded machine-readable text)
- 🖹 **OCR PDF Analysis** (scanned/handwritten documents)
- 💡 **Question-Answer Interface** powered by Google's Gemini LLM
- 🧠 **FAISS Vector Store** for contextual search and embedding-based document analysis
- 📚 Supports **handwritten**, **typed**, **scanned**, and **mixed-content** documents
- 🧾 Ideal for **business documents, handwritten notes, invoices, receipts**, and more
- 🎨 **Beautiful UI** with custom CSS styling and responsive layout

---

## 📦 Tech Stack

| Component     | Tech/Library                       |
|---------------|------------------------------------|
| 🧠 LLM         | [Google Gemini (via `google.generativeai`)](https://ai.google.dev/) |
| 🔍 Embeddings | `GoogleGenerativeAIEmbeddings` via `langchain_google_genai` |
| 📄 PDF Reading| `PyPDF` |
| 📚 Vector DB  | `FAISS` via `langchain_community.vectorstores` |
| 🧱 Text Split | `RecursiveCharacterTextSplitter` |
| 🧪 UI & UX    | `Streamlit` |
| 🖍 Styling    | Custom HTML & CSS |

---

## 🔧 Installation

### 🐍 Requirements

- Python 3.8+
- Google Gemini API key

### 💻 Setup

1. **Clone the repository**
```bash
git clone https://github.com/your-username/pdf-analysis-hub.git
cd pdf-analysis-hub
```

2. **Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the app**
```bash
streamlit run app.py
```

---

## 🔑 Get Your Gemini API Key

1. Visit [https://ai.google.dev/](https://ai.google.dev/)
2. Sign in and create a project
3. Enable **Generative Language API**
4. Copy your API key and paste it into the sidebar of the app

---

## 🧪 Usage

1. Launch the Streamlit app
2. Paste your **Gemini API key** into the sidebar
3. Choose between:
   - 📄 Normal PDF (for embedded text)
   - 🖹 OCR PDF (for scanned/handwritten documents)
4. Upload your document
5. Click **"Analyze"**
6. Ask questions in the chat interface 🤖

---

## 📁 Project Structure

```plaintext
.
├── app.py                # Main Streamlit application
├── requirements.txt      # Python dependencies
├── faiss-index/          # Saved FAISS vector store (auto-created)
├── README.md             # This file
```

---

## ⚠️ Limitations

- OCR PDFs are passed as raw binary to Gemini (no preprocessing)
- No multi-PDF search in current version
- FAISS index is stored locally and overwritten on each session

---

## ✨ Future Enhancements

- 📊 Real-time document summary generation
- 🔍 Multi-file querying
- 🗂️ Session-based vector store handling
- 📈 Dashboard analytics and metadata extraction
- 🧾 Invoice parser and entity recognition

---

## ❤️ Credits

- [LangChain](https://www.langchain.com/)
- [Google AI Gemini](https://ai.google.dev/)
- [Streamlit](https://streamlit.io/)
- [PyPDF](https://pypi.org/project/pypdf/)
- [FAISS](https://github.com/facebookresearch/faiss)

---

## 🛡 License

This project is licensed under the MIT License. See `LICENSE` for more details.

---

## 🙋‍♀️ Built With

**❤️ by Team CodeFusion**
