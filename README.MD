# AI Interviewer

An AI-powered interview platform built with Flask, OpenCV, DeepFace, and Hugging Face Transformers.

## 🌐 Live Demo & Deployment
- **Live Project URL**: [https://ai-interviewer-three.vercel.app](https://ai-interviewer-three.vercel.app) *(Deploy directly via Vercel)*
- **Vercel Configured**: Pre-configured with `vercel.json` and serverless entrypoint in `api/index.py`.

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https%3A%2F%2Fgithub.com%2Famogh-cyber%2Fai-interviewer)

---

## 🚀 Features
- **User Authentication**: Secure login and registration with role-based access (admin/user).
- **Resume Processing**: Upload resumes (PDF, DOCX, TXT) and automatically extract text.
- **Role Detection & Skill Matching**: Detects candidate target role based on resume keyword matching and calculates an ATS score.
- **Dynamic Question Generation**: Generates role-specific and resume-tailored interview questions.
- **Emotion Detection**: Uses webcam and OpenCV + DeepFace to analyze facial expressions and summarize candidate emotions in real-time.
- **Automated Answer Evaluation**: Scores user answers using NLP (TF-IDF for semantic similarity, keyword matching, and length comparison), providing deep insights and actionable feedback.

---

## 🛠️ Local Installation & Setup

### Prerequisites
- Python 3.8+
- Webcam (for real-time emotion tracking)

### Running Locally
1. Clone the repository:
   ```bash
   git clone https://github.com/amogh-cyber/ai-interviewer.git
   cd ai-interviewer
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   python app.py
   ```
   *Access the web app at `http://localhost:5000`.*

---

## ⚡ Vercel Serverless Deployment Instructions
1. Go to [Vercel Dashboard](https://vercel.com/new).
2. Connect your GitHub account and select repository **`amogh-cyber/ai-interviewer`**.
3. Keep default settings (Vercel automatically detects `vercel.json` & `api/index.py`).
4. Click **Deploy**.
5. Once deployed, copy your live `.vercel.app` URL and update the **Website** link in your GitHub repo header under **About**!