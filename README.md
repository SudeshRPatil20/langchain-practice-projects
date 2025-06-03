
# Langchain Basic Projects

Welcome to the **Langchain Basic Projects** repository! This repo contains various projects built using Langchain, GROQ API, Google APIs, and other powerful tools to build intelligent chatbots, Q&A systems, search engines, and more.

---

## 📝 Project Overview

This repository features multiple projects focused on building language model-powered applications such as:

- Chat with SQL databases
- End-to-end Question & Answer systems
- Chatbots using Langchain framework
- Math problem solvers using LLMs --- https://langchain-practice-projects-zkkzyt5rptja3x9kvfpgm5.streamlit.app/
- RAG (Retrieval Augmented Generation) systems
- Text summarization from YouTube and URLs
- Search engines integrated with agents and tools --- https://langchain-practice-projects-x9gytybk4zm7jc9wg4b4ww.streamlit.app/

Each project demonstrates different use-cases and integrations of Langchain and related APIs to help you learn and build your own intelligent apps.

---

## ⚙️ Prerequisites

Before running the projects, make sure you have:

- [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed
- Python 3.10 environment created with conda (recommended)
- `pip` package manager (comes with conda env)
- Required API keys (see next section)

---

## 🐍 Setting up a Python 3.10 Conda Environment

We recommend creating a dedicated conda environment for these projects to manage dependencies cleanly.

### Steps:

1. **Create a new conda environment with Python 3.10:**

   ```bash
   conda create -n langchain_proj python=3.10 -y


2. **Activate the environment:**

   ```bash
   conda activate langchain_proj
   ```

3. **Install required packages:**

   Make sure your `requirements.txt` is in the project root, then run:

   ```bash
   pip install -r requirements.txt
   ```

4. **Add your `.env` file with API keys** as described in the [API Keys and Setup](#-api-keys-and-setup) section.

5. **Run your Streamlit app or other scripts within this environment.**

---

### To deactivate the environment when done:

```bash
conda deactivate
```

---

## 🔑 API Keys and Setup

This project requires several API keys for full functionality. Below are the keys you will need and how to obtain them:

| API Key Name        | Purpose                                             | How to Obtain                                                                                                      |
| ------------------- | --------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| `GROQ_TESTING_API`  | Access to GROQ vector database API                  | Sign up at [GROQ](https://groq.com) and create an API key from your dashboard                                      |
| `GOOGLE_API_KEY`    | Access to Google Cloud APIs (e.g., Search, YouTube) | Create a project in [Google Cloud Console](https://console.cloud.google.com), enable APIs, and generate an API key |
| `LANGCHAIN_API_KEY` | Langchain platform API access                       | Register on [Langchain platform](https://www.langchain.com) and create a project to get the API key                |
| `LANGCHAIN_PROJECT` | Name of your Langchain project                      | Set this as the project name you create on Langchain                                                               |
| `LANGCHAIN_ID`      | (Optional) Langchain project/user ID                | Find this in your Langchain user or project dashboard                                                              |
| `HF_TOKEN`          | Hugging Face API token                              | Create an account on [Hugging Face](https://huggingface.co), go to settings → Access Tokens, generate a token      |

---

### How to add your API keys

1. Create a `.env` file in the root of your project directory.

2. Add the keys like this:

   ```
   GROQ_TESTING_API="your_groq_api_key_here"
   GOOGLE_API_KEY="your_google_api_key_here"
   LANGCHAIN_API_KEY="your_langchain_api_key_here"
   LANGCHAIN_PROJECT="your_langchain_project_name"
   LANGCHAIN_ID="your_langchain_id"
   HF_TOKEN="your_huggingface_token"
   ```

3. Make sure `.env` is added to `.gitignore` to keep your keys private.

---

## 🚀 How to run the projects

1. Clone the repo:

   ```bash
   git clone https://github.com/yourusername/langchain-basic-projects.git
   cd langchain-basic-projects
   ```

2. Set up and activate your conda environment (see above).

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Add your `.env` file with API keys as described above.

5. Run the desired project, for example:

   ```bash
   streamlit run "Math Problem Solver/app.py"
   ```



## 📂 List of Projects

| Project Name                       | Description                                      | 
| ---------------------------------- | ------------------------------------------------ | 
| Chat\_with\_SQL\_Db                | Chatbot interface that queries an SQL database   | 
| SQL with Langchain                 | Using Langchain to interact with SQL data        | 
| Chatbot using langchain            | General Q\&A chatbot using Langchain             | 
| End to End QNA                     | Complete Q\&A pipeline integrating multiple APIs | 
| LLM-with-Groq                      | Mini LangServe implementation using Groq         | 
| Math Problem Solver                | Solve math problems using language models        |
| Rag of Q\&A                        | Retrieval augmented generation for Q\&A          | 
| Search\_engine\_with\_tools\&Agent | Search engine that uses agents and tools         | 
| Text\_summarization                | Summarizes YouTube videos and URLs               | 
| conversation Q\&A chatbot          | Interactive Q\&A chatbot                         | 

---

## 📖 Additional Resources

* [Langchain Documentation](https://docs.langchain.com/)
* [GROQ API Documentation](https://docs.groq.com/)
* [Google Cloud Console](https://console.cloud.google.com/)
* [Hugging Face Documentation](https://huggingface.co/docs)

---

## 🛠 Support & Contribution

Feel free to open issues or submit pull requests if you find bugs or want to add features.

---

## 📜 License

Specify your license here.

---

**Happy coding! 🚀**

```
```
