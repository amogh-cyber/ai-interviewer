// Basic helper functions
const $ = id => document.getElementById(id);

let stream = null;
let currentRole = null;
let questions = [];
let answers = [];
let currentQuestionIndex = 0;

// Resume upload
$('upload-resume').addEventListener('click', async () => {
  const fileInput = $('resume-file');
  if (!fileInput.files.length) { alert('Choose a resume file'); return; }
  const file = fileInput.files[0];
  const form = new FormData();
  form.append('resume', file);

  const res = await fetch('/upload_resume', { method: 'POST', body: form });
  const data = await res.json();
  if (data.error) { alert('Error: ' + data.error); return; }

  currentRole = data.role;
  questions = data.questions || [];
  answers = data.answers || [];

  $('resume-result').innerText = `Role: ${data.role}\nATS: ${data.ats_score}%\nMatched: ${data.matched_skills.join(', ')}`;

  // show questions section
  $('questions-section').style.display = 'block';
  renderQuestions();
});

// render questions and answers (collapsed)
function renderQuestions() {
  const el = $('questions-list');
  el.innerHTML = '';
  questions.forEach((q, i) => {
    const div = document.createElement('div');
    div.className = 'q';
    div.innerHTML = `<strong>Q${i+1}:</strong> ${q} <button data-i="${i}" class="show-model">Show Model Answer</button>`;
    el.appendChild(div);
  });
  document.querySelectorAll('.show-model').forEach(btn => {
    btn.onclick = (ev) => {
      const i = ev.target.dataset.i;
      alert('Model answer:\n' + (answers[i] || 'N/A'));
    }
  });
}

// ------------------------ Camera ------------------------
$('start-camera').addEventListener('click', async () => {
  if (stream) return;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    $('video').srcObject = stream;
  } catch (e) {
    alert('Camera access denied or not available: ' + e);
  }
});

$('stop-camera').addEventListener('click', () => {
  if (stream) {
    stream.getTracks().forEach(t => t.stop());
    stream = null;
    $('video').srcObject = null;
  }
});

// capture current frame and send to server to analyze emotion
$('get-emotion').addEventListener('click', async () => {
  if (!stream) { alert('Start camera first'); return; }
  const video = $('video');
  const canvas = $('capture');
  canvas.width = video.videoWidth || 480;
  canvas.height = video.videoHeight || 360;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  const dataUrl = canvas.toDataURL('image/jpeg');
  const res = await fetch('/analyze_frame', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ image: dataUrl })
  });
  const data = await res.json();
  if (data.error) {
    $('emotion-result').innerText = 'Emotion error: ' + data.detail || data.error;
  } else {
    $('emotion-result').innerText = 'Dominant: ' + data.dominant + '\n' + JSON.stringify(data.emotions, null, 2);
  }
});

// ------------------------ Submit text answer + scoring ------------------------
$('submit-answer').addEventListener('click', async () => {
  const txt = $('user-answer').value;
  if (!txt || currentRole === null) { alert('Provide answer and upload resume first'); return; }
  const payload = { question_index: currentQuestionIndex, user_text: txt, role: currentRole };
  const res = await fetch('/score_answer', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(payload) });
  const data = await res.json();
  if (data.error) {
    $('score-result').innerText = 'Error scoring: ' + data.error;
  } else {
    $('score-result').innerText = `Score: ${data.score} | Similarity: ${(data.similarity*100).toFixed(1)}%`;
  }
});
