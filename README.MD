# AI Interviewer

An AI-powered interview platform built with Flask, OpenCV, DeepFace, and Hugging Face Transformers.

## Features
- **User Authentication**: Secure login and registration with role-based access (admin/user).
- **Resume Processing**: Upload resumes (PDF, DOCX, TXT) and automatically extract text.
- **Role Detection & Skill Matching**: Detects the candidate's target role based on keyword matching from their resume and calculates an ATS score.
- **Dynamic Question Generation**: Generates role-specific and resume-tailored interview questions.
- **Emotion Detection**: Uses webcam and OpenCV + DeepFace to analyze facial expressions and summarize the candidate's emotions in real-time.
- **Automated Answer Evaluation**: Scores user answers using NLP (TF-IDF for semantic similarity, keyword matching, and length comparison), providing deep insights and actionable feedback.

## Installation & Setup

### Prerequisites
- Python 3.8+
- Webcam (for emotion tracking)

### Running the App
1. Install dependencies:
   ```bash
   pip install Flask opencv-python deepface Pillow PyPDF2 python-docx transformers scikit-learn Flask-SocketIO
   ```

2. Run the application:
   ```bash
   python app.py
   ```

3. Expose the local server to the internet using ngrok (optional):
   ```bash
   ngrok http 5000
   ngrok config add-authtoken <your_token>
   ```

The database (`interview.db`) is automatically initialized upon running `app.py`.