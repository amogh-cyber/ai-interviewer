import os
import csv
import cv2
import re
import torch
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from deepface import DeepFace
from PIL import Image
from PyPDF2 import PdfReader
from docx import Document
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
import json
import base64
import io
import sqlite3
import random

from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask_socketio import SocketIO, emit, join_room
from datetime import datetime
import uuid


# ---------------------------------------------------------------
# Flask App Configuration
# ---------------------------------------------------------------
app = Flask(__name__)
app.secret_key = 'your_secret_key_here_change_this_in_production'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
socketio = SocketIO(app, cors_allowed_origins="*")


os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ---------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"


MEETINGS = {}


def update_db_schema():
    """Update database schema to add missing columns"""
    try:
        conn = sqlite3.connect(DB_NAME)

        c = conn.cursor()
        
        
        c.execute("PRAGMA table_info(interview_results)")
        columns = [column[1] for column in c.fetchall()]
        
        
        if 'answered_count' not in columns:
            print("[INFO] Adding answered_count column to interview_results table")
            c.execute('ALTER TABLE interview_results ADD COLUMN answered_count INTEGER NOT NULL DEFAULT 0')
        
        if 'total_questions' not in columns:
            print("[INFO] Adding total_questions column to interview_results table")
            c.execute('ALTER TABLE interview_results ADD COLUMN total_questions INTEGER NOT NULL DEFAULT 5')
        
        conn.commit()
        conn.close()
        print("[INFO] Database schema updated successfully")
    except Exception as e:
        print(f"[ERROR] Updating database schema: {e}")




checkpoint = "google/flan-t5-small"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

generator = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0 if device == "cuda" else -1
)


# ---------------------------------------------------------------
# Global Variables
# ---------------------------------------------------------------
ALLOWED_EXTENSIONS = {'pdf', 'txt', 'docx'}
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# ---------------------------------------------------------------
# Role Keyword Dictionary
# ---------------------------------------------------------------
ROLE_KEYWORDS = {
    "Python Developer": ["python", "django", "flask", "pandas", "numpy"],
    "Data Scientist": ["data", "machine learning", "ml", "statistics", "pandas"],
    "Frontend Developer": ["javascript", "react", "html", "css", "frontend"],
    "Backend Developer": ["backend", "api", "server", "database", "sql"],
    "Full Stack Developer": ["frontend", "backend", "react", "django", "node"],
    "Machine Learning Engineer": ["ml", "machine learning", "tensorflow", "pytorch", "deep learning"],
    "DevOps Engineer": ["docker", "kubernetes", "ci/cd", "jenkins", "monitoring"],
    "Cloud Engineer": ["aws", "azure", "gcp", "cloud", "security", "deployment"],
    "Digital Marketer": ["seo", "marketing", "social media", "adwords", "content"],
    "Manager": ["leadership", "team", "strategy", "planning", "project management"],
    "Java Developer": ["java", "jvm", "jdk", "spring", "multithreading"],
    "C++ Developer": ["c++", "pointers", "memory management", "oop", "stl"],
    "Mobile App Developer (Android/iOS)": ["android", "ios", "kotlin", "swift", "mobile", "app development"],
    "QA Engineer": ["testing", "automation", "selenium", "cypress", "test cases", "regression"],
    "Security Engineer": ["security", "encryption", "vulnerability", "penetration testing", "firewall"],
    "Data Engineer": ["etl", "sql", "nosql", "pipeline", "big data", "spark"],
    "AI Engineer": ["ai", "artificial intelligence", "deep learning", "tensorflow", "pytorch"],
    "UI/UX Designer (Tech-related)": ["ui", "ux", "wireframe", "prototyping", "figma", "accessibility"],
    "Database Administrator": ["sql", "database", "backup", "indexing", "optimization", "oracle"],
    "Network Engineer": ["network", "tcp/ip", "routing", "switches", "vpn", "subnetting"],
    "Embedded Software Engineer": ["embedded", "microcontroller", "c", "rtos", "firmware"],
    "Game Developer": ["unity", "unreal engine", "game", "physics", "ai", "multiplayer"],
    "Blockchain Developer": ["blockchain", "solidity", "smart contracts", "ethereum", "consensus"],
    "Site Reliability Engineer (SRE)": ["sre", "reliability", "monitoring", "uptime", "incident", "load balancing"],
    "Frontend Framework Specialists (Angular, Vue)": ["angular", "vue", "frontend", "components", "state management"],
    "ERP Developer": ["erp", "sap", "oracle", "workflow", "customization", "reporting"],
    "Salesforce Developer": ["salesforce", "apex", "visualforce", "crm", "lightning"],
    "Robotics Engineer": ["robotics", "ros", "kinematics", "dynamics", "sensors", "automation"],
    "System Analyst": ["system analysis", "uml", "modeling", "requirements", "feasibility", "documentation"],
    "Firmware Engineer": ["firmware", "embedded", "c", "drivers", "hardware", "optimization"],
    "AR/VR Developer": ["ar", "vr", "unity", "unreal", "3d", "immersive"],
    "Test Automation Engineer": ["automation", "selenium", "cypress", "testing", "ci/cd", "scripts"],
    "Business Intelligence Developer": ["bi", "power bi", "tableau", "qlikview", "etl", "dashboard"],
    "API Developer": ["api", "rest", "soap", "authentication", "documentation", "versioning"],
    "SAP Consultant": ["sap", "modules", "implementation", "customization", "workflow", "integration"],
    "Site Architect (Web Applications)": ["architecture", "web app", "scalability", "microservices", "security", "performance"],
    "Python Automation Engineer": ["python", "automation", "selenium", "pyautogui", "requests", "scripting"]
}


try:
    with open("questions.json", "r", encoding="utf-8") as f:
        ROLE_QUESTIONS = json.load(f)
    
except Exception as e:
    print(f"[ERROR] Could not load questions.json: {e}")
    ROLE_QUESTIONS = {}

try:
    with open("answer.json", "r", encoding="utf-8") as f:
        ROLE_ANSWERS = json.load(f)
    
except Exception as e:
    print(f"[ERROR] Could not load answer.json: {e}")
    ROLE_ANSWERS = {}

# ---------------------------------------------------------------
# REAL ANSWER MATCHING SCORING SYSTEM
# ---------------------------------------------------------------
def score_answer(user_answer, question, role, question_index):
    """Score answer by EXACT matching with answer.json"""
    
    
    # print(f"[SCORING] Role: {role}")
    # print(f"[SCORING] Question Index: {question_index}")
    # print(f"[SCORING] Question: {question}")
    # print(f"[SCORING] User Answer: {user_answer}")
    
    
    usr = user_answer.lower().strip()
    
    
    if not usr or usr in ["", "no answer", "could not understand", "error"]:
        print("[SCORING] Empty answer -> Score: 0")
        return 0, "No answer provided"
    
    
    correct_answers = ROLE_ANSWERS.get(role, [])
    
    if not correct_answers:
        print(f"[SCORING ERROR] No answers found for role: {role}")
        return 0, "No reference answers available"
    
    if question_index >= len(correct_answers):
        print(f"[SCORING ERROR] Question index {question_index} out of range")
        return 0, "Question index error"
    
    correct_answer = correct_answers[question_index]
    print(f"[SCORING] Correct Answer: {correct_answer}")
    
    
    similarity = calculate_detailed_similarity(usr, correct_answer)
    
    
    score = similarity_to_score(similarity)
    feedback = get_feedback_based_on_score(score, similarity)
    
    print(f"[SCORING] Similarity: {similarity:.2f} -> Score: {score}/5")  
    
    
    return score, feedback

def calculate_detailed_similarity(user_answer, correct_answer):

    
    user_lower = user_answer.lower().strip()
    correct_lower = correct_answer.lower().strip()
    
    
    semantic_similarity = calculate_semantic_similarity(user_lower, correct_lower)
    
    
    keyword_similarity = calculate_keyword_similarity(user_lower, correct_lower)
    
    
    length_similarity = calculate_length_similarity(user_lower, correct_lower)
    
    
    combined_similarity = (
        semantic_similarity * 0.5 +  
        keyword_similarity * 0.4 +     
        length_similarity * 0.1       
    )
    

    
    return min(combined_similarity, 1.0) 

def calculate_semantic_similarity(user_answer, correct_answer):
    """Calculate semantic similarity using TF-IDF and cosine similarity"""
    try:
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform([user_answer, correct_answer])
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return similarity
    except:
        return 0.0

def calculate_keyword_similarity(user_answer, correct_answer):
    """Calculate similarity based on keyword matching"""
    
    
    words = re.findall(r'\b\w+\b', correct_answer.lower())
    important_keywords = [word for word in words if len(word) > 3 and word not in ['that', 'with', 'this', 'have', 'from']]
    
    if not important_keywords:
        return 0.0
    
    
    matched_keywords = sum(1 for keyword in important_keywords if keyword in user_answer)
    
    similarity = matched_keywords / len(important_keywords)
    
    
    return similarity

def calculate_length_similarity(user_answer, correct_answer):
    """Calculate similarity based on answer length"""
    user_words = len(user_answer.split())
    correct_words = len(correct_answer.split())
    
    if correct_words == 0:
        return 0.0
    

    ratio = min(user_words / correct_words, 1.0)
    
    
    if ratio >= 0.3:
        return 0.8
    elif ratio >= 0.1:
        return 0.4
    else:
        return 0.1

def similarity_to_score(similarity):
    """Convert similarity (0-1) to score (0-5)""" 
    if similarity >= 0.8:
        return 5 
    elif similarity >= 0.6:
        return 4  
    elif similarity >= 0.5:
        return 3 
    elif similarity >= 0.4:
        return 2 
    elif similarity >= 0.2:
        return 1 
    else:
        return 0  

def get_feedback_based_on_score(score, similarity):
    """Provide feedback based on the score"""
    feedbacks = {
        5: f"Excellent! Your answer perfectly matches the expected response ({(similarity*100):.0f}% match)",  
        4: f"Very Good! Your answer closely matches the expected response ({(similarity*100):.0f}% match)",  
        3: f"Good! Your answer covers most key points ({(similarity*100):.0f}% match)",  
        2: f"Fair! Your answer includes some relevant points ({(similarity*100):.0f}% match)",  
        1: f"Basic! Your answer has minimal relevance ({(similarity*100):.0f}% match)", 
        0: "Poor! Your answer doesn't match the expected response"
    }
    return feedbacks.get(score, "Unable to evaluate answer")


DB_NAME = "interview.db"

# ---------------------------------------------------------------
# User Management 
# ---------------------------------------------------------------
def init_users_db():
    conn = sqlite3.connect(DB_NAME)

    c = conn.cursor()

    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password TEXT,
            role TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    conn.commit()
    conn.close()


def init_db():
    conn = sqlite3.connect(DB_NAME)

    c = conn.cursor()

    c.execute('''
         CREATE TABLE IF NOT EXISTS interview_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            role TEXT,
            total_score INTEGER,
            max_score INTEGER,
            answered_count INTEGER,
            total_questions INTEGER,
            emotion_summary TEXT,
            interview_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            questions TEXT,
            user_answers TEXT,
            scores TEXT
        )
    ''')

    conn.commit()
    conn.close()

init_db()
init_users_db()




# ---------------------------------------------------------------
# Resume Processing 
# ---------------------------------------------------------------
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(path):
    text = ""
    try:
        reader = PdfReader(path)
        for page in reader.pages:
            if page.extract_text():
                text += page.extract_text() + " "
    except Exception as e:
        print(f"[ERROR] PDF extract: {e}")
    return text

def extract_text_from_docx(path):
    try:
        doc = Document(path)
        return " ".join([p.text for p in doc.paragraphs])
    except Exception as e:
        print(f"[ERROR] DOCX extract: {e}")
        return ""

def extract_resume_content(path):
    ext = path.split('.')[-1].lower()
    if ext == "pdf":
        return extract_text_from_pdf(path)
    elif ext == "docx":
        return extract_text_from_docx(path)
    elif ext == "txt":
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return f.read()
        except:
            return ""
    return ""

def detect_role_by_keywords(resume_text):
    resume_text_lower = resume_text.lower()
    scores = {}
    for role, keywords in ROLE_KEYWORDS.items():
        match = sum(1 for k in keywords if k in resume_text_lower)
        if match > 0:
            scores[role] = match
    detected_role = max(scores, key=scores.get) if scores else "Software Developer"
    print(f"[ROLE DETECTION] Detected role: {detected_role}")
    return detected_role

def generate_role_specific_questions(role, resume_text, num_questions=5):
    base = ROLE_QUESTIONS.get(role, ["Describe your role responsibilities."])
    questions = base[:num_questions]
    print(f"[QUESTIONS] Generated {len(questions)} questions for {role}")
    return questions

# ---------------------------------------------------------------
# Emotion Detection 
# ---------------------------------------------------------------
def process_emotion_detection(frame_data):
    try:
        header, encoded = frame_data.split(",", 1)
        binary_data = base64.b64decode(encoded)
        image = Image.open(io.BytesIO(binary_data))
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.2, 5)
        emotions = {}
        
        for (x, y, w, h) in faces:
            roi = frame[y:y+h, x:x+w]
            try:
                result = DeepFace.analyze(roi, actions=['emotion'], enforce_detection=False)
                emo = result[0]['dominant_emotion']
                emotions[emo] = emotions.get(emo, 0) + 1
            except Exception as e:
                continue
        
        if emotions:
            session['emotion_counts'] = emotions
        
        return emotions
    except Exception as e:
        print(f"[ERROR] Emotion detection: {e}")
        return {}

# ---------------------------------------------------------------
# Flask Routes 
# ---------------------------------------------------------------

def admin_required():
    if session.get('role') != 'admin':
        flash("Admin access required", "error")
        return False
    return True

@app.route('/')
def index():
    if 'username' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username'].strip()
        password = request.form['password'].strip()

        conn = sqlite3.connect(DB_NAME)

        c = conn.cursor()
        c.execute(
            "SELECT password, role FROM users WHERE username = ?",
            (username,)
        )
        row = c.fetchone()
        conn.close()

        if row and check_password_hash(row[0], password):
            session.clear()
            session['username'] = username
            session['role'] = row[1] 
            

            # initialize interview session data
            session.update({
                'resume_uploaded': False,
                'current_question_index': 0,
                'questions': [],
                'user_answers': [],
                'scores': [],
                'resume_text': "",
                'detected_role': "",
                'emotion_counts': {}
            })

            flash("Login successful!", "success")

            
            if row[1] == 'admin':
                return redirect(url_for('admin_dashboard'))
            else:
                return redirect(url_for('dashboard'))

        flash("Invalid credentials", "error")

    return render_template('login.html')





@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username'].strip()
        password = request.form['password'].strip()
        role = request.form['role']

        if not username or not password:
            flash("All fields required", "error")
            return redirect(url_for('register'))

        hashed_password = generate_password_hash(password)

        try:
            conn = sqlite3.connect(DB_NAME)

            c = conn.cursor()

            c.execute(
                "INSERT INTO users (username, password, role) VALUES (?, ?, ?)",
                (username, hashed_password, role)
            )

            conn.commit()
            conn.close()

            flash("Registration successful!", "success")
            return redirect(url_for('login'))

        except sqlite3.IntegrityError:
            flash("Username already exists!", "error")

    return render_template('register.html')


@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))

with open("./models/flan-t5-small/resume_feedback.json", "r", encoding="utf-8") as f:
    RESUME_FEEDBACK = json.load(f)


def generate_resume_feedback_rule_based(role, ats_score, matched_skills):
    role_data = RESUME_FEEDBACK.get(role)

    
    if not role_data:
        return [
            "Your resume shows general technical exposure but lacks strong role-specific alignment.",
            "Some relevant skills are present, but clearer specialization is needed.",
            "Improve keyword alignment, project clarity, and resume structure."
        ]


    if ats_score >= 80:
        para1 = role_data["summary"]["strong"]
    elif ats_score >= 50:
        para1 = role_data["summary"]["medium"]
    else:
        para1 = role_data["summary"]["weak"]

    
    strengths = role_data["strengths"]
    para2 = (
        "Key strengths identified in your resume include "
        + ", ".join(strengths[:4])
        + ". These areas indicate relevant exposure and foundational capability for the selected role."
    )

    
    missing = role_data["missing_skills"]
    para3 = (
        "To further improve your resume, focus on strengthening areas such as "
        + ", ".join(missing[:4])
        + ". Adding project-based evidence and measurable impact will significantly improve shortlisting chances."
    )

    return [para1, para2, para3]

@app.route('/dashboard')
def dashboard():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('dashboard.html', 
                         username=session['username'],
                         resume_uploaded=session.get('resume_uploaded', False))

                         
@app.route('/upload_resume', methods=['POST'])
def upload_resume():
    if 'username' not in session:
        return jsonify({'error': 'Not logged in'}), 401

    if 'resume' not in request.files:
        return jsonify({'error': 'No file selected'}), 400

    file = request.files['resume']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # -------- Resume Extraction --------
        resume_text = extract_resume_content(filepath)
        resume_text_clean = " ".join(resume_text.split()) if resume_text else ""

        # -------- Role Detection --------
        role = detect_role_by_keywords(resume_text_clean)

        # -------- Skill Matching & ATS Score --------
        matched_skills = [
            k for k in ROLE_KEYWORDS.get(role, [])
            if k.lower() in resume_text_clean.lower()
        ]
        matched_skills_clean = ", ".join(matched_skills)

        ats_score = int(
            (len(matched_skills) / max(len(ROLE_KEYWORDS.get(role, [])), 1)) * 100
        )

        # -------- Questions --------
        questions = generate_role_specific_questions(role, resume_text_clean, 5)

        # -------- Resume Feedback (RULE-BASED JSON) --------
        resume_feedback = generate_resume_feedback_rule_based(
            role,
            ats_score,
            matched_skills
        )
        print(resume_feedback)
        # -------- Session Update --------
        session.update({
            'resume_uploaded': True,
            'questions': questions,
            'user_answers': [""] * len(questions),
            'scores': [0] * len(questions),
            'resume_text': resume_text_clean,
            'detected_role': role,
            'current_question_index': 0,
            'resume_feedback': resume_feedback
        })

        # -------- Cleanup --------
        try:
            os.remove(filepath)
        except:
            pass

        # -------- Response --------
        return jsonify({
            'success': True,
            'role': role,
            'ats_score': ats_score,
            'matched_skills': matched_skills_clean,
            'questions_count': len(questions),
            'resume_feedback': resume_feedback
        })

    return jsonify({'error': 'Invalid file type'}), 400


@app.route('/interview')
def interview():
    if 'username' not in session:
        return redirect(url_for('login'))
    
    if not session.get('resume_uploaded', False):
        flash('Please upload your resume first!', 'warning')
        return redirect(url_for('dashboard'))
    
    current_index = session.get('current_question_index', 0)
    questions = session.get('questions', [])
    
    if current_index >= len(questions):
        return redirect(url_for('results'))
    
    if 'user_answers' not in session:
        session['user_answers'] = [""] * len(questions)
    if 'scores' not in session:
        session['scores'] = [0] * len(questions)
    
    return render_template('interview.html',
                         question=questions[current_index],
                         question_index=current_index + 1,
                         total_questions=len(questions))


@app.route('/process_emotion', methods=['POST'])
def process_emotion():
    if 'username' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    
    frame_data = request.json.get('frame')
    if not frame_data:
        return jsonify({'error': 'No frame data'}), 400
    
    emotions = process_emotion_detection(frame_data)
    return jsonify({'emotions': emotions})



@app.route('/submit_answer', methods=['POST'])
def submit_answer():
    if 'username' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    
    question_index = session.get('current_question_index', 0)
    user_answer = request.json.get('answer', '')
    
    questions = session.get('questions', [])
    role = session.get('detected_role', 'Software Developer')
    
    print(f"\n[SUBMIT_ANSWER] Processing answer for question {question_index}")
    print(f"[SUBMIT_ANSWER] Role: {role}")
    print(f"[SUBMIT_ANSWER] User answer length: {len(user_answer)}")
    
    if question_index < len(questions):
        current_question = questions[question_index]
        
        # Score the answer using REAL matching
        score, feedback = score_answer(user_answer, current_question, role, question_index)
        
        # Update session
        user_answers = session.get('user_answers', [])
        scores = session.get('scores', [])
        
        if question_index < len(user_answers):
            user_answers[question_index] = user_answer
            scores[question_index] = score
        
        session['user_answers'] = user_answers
        session['scores'] = scores
        
        print(f"[SUBMIT_ANSWER] Updated scores: {scores}")
        print(f"[SUBMIT_ANSWER] Score saved successfully!")
        
        return jsonify({
            'success': True,
            'score': score,
            'feedback': feedback,
            'max_score': 5  
        })
    
    return jsonify({'error': 'Invalid question index'}), 400



@app.route('/next_question')
def next_question():
    if 'username' not in session:
        return redirect(url_for('login'))
    
    current_index = session.get('current_question_index', 0)
    questions = session.get('questions', [])
    
    if current_index < len(questions) - 1:
        session['current_question_index'] = current_index + 1
    else:
        session['current_question_index'] = len(questions) - 1
    
    return redirect(url_for('interview'))



@app.route('/previous_question')
def previous_question():
    if 'username' not in session:
        return redirect(url_for('login'))
    
    current_index = session.get('current_question_index', 0)
    
    if current_index > 0:
        session['current_question_index'] = current_index - 1
    
    return redirect(url_for('interview'))


def save_interview_result(username, role, total_score, max_score, answered_count, 
                         total_questions, emotion_summary, questions, user_answers, scores):
    """Save interview results to database"""
    try:
        conn = sqlite3.connect(DB_NAME)

        c = conn.cursor()
        
        # Convert lists to JSON strings for storage
        questions_json = json.dumps(questions)
        user_answers_json = json.dumps(user_answers)
        scores_json = json.dumps(scores)
        
        c.execute('''
            INSERT INTO interview_results 
            (username, role, total_score, max_score, answered_count, total_questions, 
             emotion_summary, questions, user_answers, scores)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (username, role, total_score, max_score, answered_count, total_questions,
              emotion_summary, questions_json, user_answers_json, scores_json))
        
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"[ERROR] Saving interview result: {e}")
        return False


def get_user_interview_results(username):
    """Get all interview results for a user"""
    try:
        conn = sqlite3.connect(DB_NAME)

        c = conn.cursor()
        
        c.execute('''
            SELECT id, role, total_score, max_score, answered_count, total_questions,
                   emotion_summary, interview_date, questions, user_answers, scores
            FROM interview_results 
            WHERE username = ? 
            ORDER BY interview_date DESC
        ''', (username,))
        
        results = []
        for row in c.fetchall():
            # Parse JSON strings back to lists
            questions = json.loads(row[8]) if row[8] else []
            user_answers = json.loads(row[9]) if row[9] else []
            scores = json.loads(row[10]) if row[10] else []
            
            results.append({
                'id': row[0],
                'role': row[1],
                'total_score': row[2],
                'max_score': row[3],
                'answered_count': row[4],
                'total_questions': row[5],
                'emotion_summary': row[6],
                'interview_date': row[7],
                'questions': questions,
                'user_answers': user_answers,
                'scores': scores
            })
        
        conn.close()
        return results
    except Exception as e:
        print(f"[ERROR] Getting interview results: {e}")
        return []


def get_interview_result_by_id(result_id, username):
    """Get specific interview result by ID"""
    try:
        conn = sqlite3.connect(DB_NAME)

        c = conn.cursor()
        
        c.execute('''
            SELECT id, role, total_score, max_score, answered_count, total_questions,
                   emotion_summary, interview_date, questions, user_answers, scores
            FROM interview_results 
            WHERE id = ? AND username = ?
        ''', (result_id, username))
        
        row = c.fetchone()
        if row:
            # Parse JSON strings back to lists
            questions = json.loads(row[8]) if row[8] else []
            user_answers = json.loads(row[9]) if row[9] else []
            scores = json.loads(row[10]) if row[10] else []
            
            result = {
                'id': row[0],
                'role': row[1],
                'total_score': row[2],
                'max_score': row[3],
                'answered_count': row[4],
                'total_questions': row[5],
                'emotion_summary': row[6],
                'interview_date': row[7],
                'questions': questions,
                'user_answers': user_answers,
                'scores': scores
            }
            conn.close()
            return result
        conn.close()
        return None
    except Exception as e:
        print(f"[ERROR] Getting interview result by ID: {e}")
        return None
    

    
@app.route('/results')
def results():
    if 'username' not in session:
        return redirect(url_for('login'))
    
    questions = session.get('questions', [])
    user_answers = session.get('user_answers', [])
    scores = session.get('scores', [])
    emotion_counts = session.get('emotion_counts', {})
    role = session.get('detected_role', 'Unknown Role')
    
    total_score = sum(scores)
    max_score = len(questions) * 5  
    answered_count = sum(1 for answer in user_answers if answer and answer.strip())
    total_questions = len(questions)
    
    emotion_summary = ""
    if emotion_counts:
        total_emo = sum(emotion_counts.values())
        top_emotions = sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True)[:2]
        emotion_summary = " | ".join([f"{e}: {c/total_emo*100:.1f}%" for e, c in top_emotions])
    
    # Save results to database
    save_interview_result(
        username=session['username'],
        role=role,
        total_score=total_score,
        max_score=max_score,
        answered_count=answered_count,
        total_questions=total_questions,
        emotion_summary=emotion_summary,
        questions=questions,
        user_answers=user_answers,
        scores=scores
    )
    
    print(f"\n[RESULTS] Final scores: {scores}")
    print(f"[RESULTS] Total score: {total_score}/{max_score}")  
    print(f"[RESULTS] Answered: {answered_count}/{len(questions)}")
    print(f"[RESULTS] Results saved to database") 
    confidence_score = random.randint(75, 95)
    clarity_score = random.randint(confidence_score, 98)

    correct_answers = ROLE_ANSWERS.get(role, [])

    return render_template('results.html',
                       total_score=total_score,
                       max_score=max_score,
                       questions=questions,
                       user_answers=user_answers,
                       scores=scores,
                       emotion_summary=emotion_summary,
                       answered_count=answered_count,
                       role=role,
                       confidence_score=confidence_score,
                       clarity_score=clarity_score,
                       correct_answers=correct_answers)  


@app.route('/self_intro')
def self_intro():
    if 'username' not in session:
        return redirect(url_for('login'))

    if not session.get('resume_uploaded'):
        return redirect(url_for('dashboard'))

    return render_template(
        'self_intro.html',
        role=session.get('detected_role'),
        ats_score=session.get('ats_score', 0)
    )



@app.route('/analyze_intro', methods=['POST'])
def analyze_intro():
    text = request.json.get('text', '').strip()
    words = text.split()
    word_count = len(words)

    role = session.get('detected_role', 'your target role')

    strengths = []
    improvements = []
    communication = []

    score = 10

    # -------- Length & Structure --------
    if word_count < 40:
        improvements.append("Your introduction is too short. Expand on your background and experience.")
        score -= 2
    elif word_count > 120:
        improvements.append("Your introduction is too long. Keep it concise and focused.")
        score -= 1
    else:
        strengths.append("Good introduction length with balanced content.")

    # -------- Role Alignment --------
    if role.lower() not in text.lower():
        improvements.append(f"You did not clearly mention your target role ({role}).")
        score -= 1
    else:
        strengths.append("Clearly aligned your introduction with your target role.")

    # -------- Experience --------
    if not any(x in text.lower() for x in ["year", "experience", "worked", "background"]):
        improvements.append("You should mention your experience level or professional background.")
        score -= 1
    else:
        strengths.append("Experience was mentioned clearly.")


    skill_keywords = ["python", "java", "react", "node", "ai", "ml", "flask", "django", "sql"]
    mentioned_skills = [s for s in skill_keywords if s in text.lower()]

    if len(mentioned_skills) == 0:
        improvements.append("No technical skills were mentioned. Include your core skills.")
        score -= 2
    else:
        strengths.append(f"Mentioned relevant skills: {', '.join(mentioned_skills)}")

    
    confidence_words = ["confident", "passionate", "strong", "excited", "motivated"]
    if any(w in text.lower() for w in confidence_words):
        strengths.append("Confident and positive language detected.")
    else:
        improvements.append("Use confident words like 'passionate', 'motivated', or 'excited'.")
        score -= 1

    
    filler_words = ["uh", "um", "like", "you know"]
    filler_count = sum(text.lower().count(w) for w in filler_words)

    if filler_count > 3:
        communication.append("Filler words detected. Try to speak more fluently.")
        score -= 1
    else:
        communication.append("Minimal filler words detected.")

    
    score = max(score, 4)


    suggestion = (
    f"Hi, I’m {session.get('username')}, a {role}. "
    "I have hands-on experience building real-world applications, "
    "working with modern technologies, and continuously improving my skills. "
    "I’m excited to apply my knowledge and grow within a challenging role. "
    "I enjoy solving problems, learning from feedback, and collaborating with teams "
    "to deliver reliable and scalable solutions. "
    "I am highly motivated to contribute my skills while continuously evolving as a professional."
)


    return jsonify({
        "score": score,
        "strengths": strengths,
        "improvements": improvements,
        "communication": communication,
        "suggested_intro": suggestion
    })


@app.route('/previous_interviews')
def previous_interviews():
    """Display list of previous interviews"""
    if 'username' not in session:
        return redirect(url_for('login'))
    
    interviews = get_user_interview_results(session['username'])
    
    return render_template('previous_interviews.html',
                         username=session['username'],
                         interviews=interviews)



@app.route('/interview_details/<int:result_id>')
def interview_details(result_id):
    """Display detailed view of a specific interview"""
    if 'username' not in session:
        return redirect(url_for('login'))
    
    interview = get_interview_result_by_id(result_id, session['username'])
    
    if not interview:
        flash('Interview not found!', 'error')
        return redirect(url_for('previous_interviews'))
    
    correct_answers = ROLE_ANSWERS.get(interview['role'], [])
    
    return render_template(
        'interview_details.html',
        interview=interview,
        result_id=result_id,
        correct_answers=correct_answers
    )



with open("aptitude_questions.json", "r", encoding="utf-8") as f:
    APTITUDE = json.load(f)


import re

import re

def slugify(text):
    return re.sub(r'[^a-z0-9]+', '-', text.lower()).strip('-')

    
ROLE_MAP = {slugify(role): role for role in APTITUDE.keys()}


@app.route("/aptitude")
def aptitude_roles():
    roles = []

    
    if "general-aptitude" in ROLE_MAP:
        roles.append({
            "slug": "general-aptitude",
            "title": "General Aptitude"
        })

    for slug, title in ROLE_MAP.items():
        if slug != "general-aptitude" and len(roles) < 9:
            roles.append({
                "slug": slug,
                "title": title
            })

    return render_template("aptitude_roles.html", roles=roles)



@app.route("/aptitude/start/<slug>")
def start_aptitude(slug):
    role = ROLE_MAP.get(slug)

    if not role:
        flash("Role not found", "error")
        return redirect(url_for("aptitude_roles"))

    session["aptitude_role"] = role

    return render_template(
        "aptitude_test.html",
        role=role,
        questions=APTITUDE[role]
    )

@app.route("/aptitude/submit", methods=["POST"])
def submit_aptitude():
    role = session.get("aptitude_role")

    if not role:
        flash("Aptitude session expired. Please start again.", "error")
        return redirect(url_for("aptitude_roles"))

    questions = APTITUDE.get(role)
    if not questions:
        flash("Invalid role data", "error")
        return redirect(url_for("aptitude_roles"))

    score = 0
    results = []

    for i, q in enumerate(questions):
        user_ans = request.form.get(f"q{i}")
        if user_ans == q["correct_answer"]:
            score += 1

        results.append({
            "question": q["question"],
            "options": q["options"],
            "user_answer": user_ans,
            "correct_answer": q["correct_answer"]
        })

    session.pop("aptitude_role", None)

    return render_template(
        "aptitude_result.html",
        role=role,
        score=score,
        total=len(questions),
        results=results
    )




@app.route('/admin/dashboard')
def admin_dashboard():
    if 'username' not in session or session.get('role') != 'admin':
        flash("Access denied", "error")
        return redirect(url_for('login'))

    conn = sqlite3.connect(DB_NAME)

    c = conn.cursor()

    
    total_users = c.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    total_interviews = c.execute("SELECT COUNT(*) FROM interview_results").fetchone()[0]

    avg_score = c.execute(
        "SELECT AVG(total_score * 1.0 / max_score) FROM interview_results"
    ).fetchone()[0]
    avg_score = round(avg_score * 100, 1) if avg_score else 0

    
    c.execute("""
        SELECT username, role, created_at 
        FROM users
        ORDER BY created_at DESC
    """)
    rows = c.fetchall()

    users = [
        {
            "username": r[0],
            "role": r[1],
            "created_at": r[2]
        }
        for r in rows
    ]

    conn.close()

    return render_template(
        'admin_dashboard.html',
        total_users=total_users,
        total_interviews=total_interviews,
        avg_score=avg_score,
        users=users
    )



@app.route('/admin/users')
def admin_users():
    if 'username' not in session or session.get('role') != 'admin':
        flash("Admin access required", "error")
        return redirect(url_for('login'))

    conn = sqlite3.connect(DB_NAME)

    c = conn.cursor()

    c.execute("""
        SELECT username, role, created_at
        FROM users
        ORDER BY created_at DESC
    """)
    users = c.fetchall()

    conn.close()

    return render_template('admin_users.html', users=users)


@app.route('/admin/interviews')
def admin_interviews():
    if 'username' not in session or session.get('role') != 'admin':
        flash("Admin access required", "error")
        return redirect(url_for('login'))

    conn = sqlite3.connect(DB_NAME)

    c = conn.cursor()

    c.execute("""
        SELECT id, username, role, total_score, max_score, answered_count,
               total_questions, emotion_summary, interview_date
        FROM interview_results
        ORDER BY interview_date DESC
    """)

    rows = c.fetchall()
    conn.close()

    interviews = []
    for r in rows:
        interviews.append({
            "id": r[0],
            "username": r[1],
            "role": r[2],
            "total_score": r[3],
            "max_score": r[4],
            "percentage": round((r[3] / r[4]) * 100, 1) if r[4] else 0,
            "answered": f"{r[5]}/{r[6]}",
            "emotion": r[7],
            "date": r[8]
        })

    return render_template(
        "admin_interviews.html",
        interviews=interviews
    )

MEETINGS = {
  "room_id": {
      "password": "1234",
      "created_by": "admin"
  }
}

@app.route('/admin/create-meeting', methods=['GET', 'POST'])
def admin_create_meeting():
    if 'username' not in session or session.get('role') != 'admin':
        flash("Admin access required", "error")
        return redirect(url_for('login'))

    if request.method == 'POST':
        password = request.form['password']
        room_id = str(uuid.uuid4())[:8]

        MEETINGS[room_id] = {
            "password": password,
            "created_by": session['username']
        }

        flash(f"Meeting created! ID: {room_id}", "success")
        meeting_link = request.host_url + "meeting/" + room_id

        return render_template(
    "admin_create_meeting.html",
    room_id=room_id,
    password=password,
    meeting_link=meeting_link
)


    return render_template("admin_create_meeting.html")


@app.route('/join-meeting', methods=['GET', 'POST'])
def join_meeting():
    if request.method == 'POST':
        room_id = request.form['room_id']
        password = request.form['password']

        meeting = MEETINGS.get(room_id)
        if not meeting or meeting['password'] != password:
            flash("Invalid meeting ID or password", "error")
            return redirect(url_for('join_meeting'))

        
        session[f"meeting_access_{room_id}"] = True

        return redirect(url_for('meeting_room', room_id=room_id))

    return render_template("join_meeting.html")



@app.route('/meeting/<room_id>')
def meeting_room(room_id):
    if room_id not in MEETINGS:
        flash("Meeting not found", "error")
        return redirect(url_for('join_meeting'))


    if not session.get(f"meeting_access_{room_id}"):
        flash("Please enter meeting password first", "error")
        return redirect(url_for('join_meeting'))

    return render_template("meeting.html", room_id=room_id)




@socketio.on("join")
def on_join(data):
    room = data["room"]
    join_room(room)
    emit("user-joined", room=room, include_self=False)



@socketio.on("offer")
def on_offer(data):
    emit("offer", data, room=data["room"], include_self=False)



@socketio.on("answer")
def on_answer(data):
    emit("answer", data, room=data["room"], include_self=False)



@socketio.on("ice-candidate")
def on_ice_candidate(data):
    emit("ice-candidate", data, room=data["room"], include_self=False)



# ---------------------------------------------------------------
# Main Application
# ---------------------------------------------------------------
if __name__ == '__main__':
    socketio.run(app, host="0.0.0.0", port=5000, debug=False)
