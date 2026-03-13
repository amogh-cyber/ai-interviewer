# ============================================================
# STEP 0: INSTALL REQUIREMENTS (RUN ONCE)
# pip install transformers datasets torch sentencepiece accelerate
# ============================================================


# ============================================================
# STEP 1: IMPORTS
# ============================================================
import json
import os

from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    TrainingArguments,
    Trainer
)

# ============================================================
# STEP 2: FILE PATHS (YOUR EXISTING JSON FILES)
# ============================================================
QUESTIONS_FILE = "questions.json"
ANSWERS_FILE = "answer.json"
RESUME_FEEDBACK_FILE = "./models/flan-t5-small/resume_feedback.json"
APTITUDE_FILE = "aptitude_questions.json"

OUTPUT_MODEL_DIR = "./fine_tuned_model"

# ============================================================
# STEP 3: LOAD ALL JSON FILES
# ============================================================
with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
    QUESTIONS = json.load(f)

with open(ANSWERS_FILE, "r", encoding="utf-8") as f:
    ANSWERS = json.load(f)

with open(RESUME_FEEDBACK_FILE, "r", encoding="utf-8") as f:
    RESUME_FEEDBACK = json.load(f)

with open(APTITUDE_FILE, "r", encoding="utf-8") as f:
    APTITUDE = json.load(f)

print(" All JSON files loaded")

# ============================================================
# STEP 4: CREATE TRAINING DATA (INPUT → OUTPUT)
# ============================================================
training_samples = []

# ---------- INTERVIEW QUESTIONS ----------
for role, questions in QUESTIONS.items():
    answers = ANSWERS.get(role, [])

    for i in range(min(len(questions), len(answers))):
        training_samples.append({
            "input": f"Interview question for {role}: {questions[i]}",
            "output": answers[i]
        })

# ---------- RESUME FEEDBACK ----------
for role, data in RESUME_FEEDBACK.items():
    training_samples.append({
        "input": f"Give strong resume feedback for {role}",
        "output": data["summary"]["strong"]
    })
    training_samples.append({
        "input": f"Give weak resume feedback for {role}",
        "output": data["summary"]["weak"]
    })

# ---------- APTITUDE QUESTIONS ----------
for role, questions in APTITUDE.items():
    for q in questions:
        training_samples.append({
            "input": f"Aptitude question: {q['question']}",
            "output": q["correct_answer"]
        })

print(f" Training samples created: {len(training_samples)}")

# ============================================================
# STEP 5: CONVERT TO HUGGINGFACE DATASET
# ============================================================
dataset = Dataset.from_list(training_samples)

# ============================================================
# STEP 6: LOAD BASE MODEL (FLAN-T5)
# ============================================================
MODEL_NAME = "google/flan-t5-small"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

print(" Base FLAN-T5 model loaded")

# ============================================================
# STEP 7: TOKENIZATION
# ============================================================
def preprocess(example):
    inputs = tokenizer(
        example["input"],
        max_length=256,
        truncation=True,
        padding="max_length"
    )

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            example["output"],
            max_length=256,
            truncation=True,
            padding="max_length"
        )

    inputs["labels"] = labels["input_ids"]
    return inputs

tokenized_dataset = dataset.map(preprocess, remove_columns=["input", "output"])

print(" Dataset tokenized")

# ============================================================
# STEP 8: TRAINING CONFIGURATION (CPU SAFE)
# ============================================================
training_args = TrainingArguments(
    output_dir=OUTPUT_MODEL_DIR,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-5,
    logging_steps=50,
    save_steps=500,
    save_total_limit=2,
    fp16=False,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset
)

# ============================================================
# STEP 9: START FINE-TUNING
# ============================================================
print("Fine-tuning started...")
trainer.train()

# ============================================================
# STEP 10: SAVE MODEL
# ============================================================
trainer.save_model(OUTPUT_MODEL_DIR)
tokenizer.save_pretrained(OUTPUT_MODEL_DIR)

print(" Fine-tuning completed & model saved")

# ============================================================
# STEP 11: TEST THE MODEL (OPTIONAL)
# ============================================================
def test_model(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=150)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

print("\nTEST OUTPUT:")
print(test_model("Interview question for Python Developer: What is Flask?"))

