import os
import re

from flask import Flask, jsonify, render_template, request, send_from_directory

def read_api_key():
    env_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if env_key:
        return env_key
    key_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "api_key.txt")
    if os.path.exists(key_file):
        with open(key_file) as f:
            return f.read().strip()
    return ""

API_KEY = read_api_key()
MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")

if API_KEY:
    from google import genai
    from google.genai import types as genai_types

    client = genai.Client(api_key=API_KEY)
else:
    client = None

app = Flask(__name__, static_folder="static")

COURSES = [
    "B.Tech Computer Science and Engineering",
    "B.Tech Information Technology",
    "B.Tech Mechanical Engineering",
    "B.Tech Civil Engineering",
    "B.Tech CSE (Artificial Intelligence and Machine Learning)",
    "B.Tech CSE (Data Science)",
    "B.Tech CSE (Internet of Things)",
    "B.Tech CSE (Cyber Security)",
    "B.Tech CSE",
]

SYSTEM_PROMPT = (
    "You are Shadow, the official AI assistant of Rayat Bahra Institute of Engineering "
    "and Nano Technology, Hoshiarpur campus. Answer questions about admissions, courses, "
    "fees, scholarships, exams, timetables, campus facilities, and student services at this "
    "institute. Keep answers short and friendly. If you don't know, say you'll connect them "
    "with the college office.\n\n"
    "Courses offered at the institute:\n"
    + "\n".join("- " + c for c in COURSES)
)

# Offline fallback answers used only when no API key is set or the API fails.
OFFLINE_ANSWERS = {
    "admission": "Admissions are open at Rayat Bahra Institute of Engineering and Nano Technology, Hoshiarpur! Applications are accepted on the institute website.",
    "course": "We offer B.Tech in Computer Science and Engineering, IT, Mechanical, Civil, as well as CSE specializations in AI & Machine Learning, Data Science, IoT, and Cyber Security.",
    "btech": "We offer B.Tech in Computer Science and Engineering, IT, Mechanical, Civil, as well as CSE specializations in AI & Machine Learning, Data Science, IoT, and Cyber Security.",
    "branch": "We offer B.Tech in Computer Science and Engineering, IT, Mechanical, Civil, as well as CSE specializations in AI & Machine Learning, Data Science, IoT, and Cyber Security.",
    "fee": "Fee details are available on the institute portal under 'Fee Structure', or visit the accounts office at the Hoshiarpur campus.",
    "scholarship": "There are merit-based and need-based scholarships at Rayat Bahra. Contact the student services office for forms.",
    "exam": "Midterm exams start next month. Check your timetable on the student portal for exact dates.",
    "library": "The library is open Monday-Saturday from 8 AM to 8 PM. A student ID is required to issue books.",
    "hostel": "Hostel rooms are available on a first-come basis. Apply through the hostel office at the Hoshiarpur campus by next Friday.",
    "canteen": "The canteen is open from 8 AM to 6 PM and serves breakfast, lunch, and snacks.",
    "placement": "The placement cell conducts campus drives every semester. Register your resume on the portal.",
}


def offline_reply(message):
    text = message.lower()
    for keyword, reply in OFFLINE_ANSWERS.items():
        if re.search(rf"\b{re.escape(keyword)}s?\b", text):
            return reply
    return "I can help with admissions, fees, exams, scholarships, hostel, library, canteen, and placements."


def build_conversation(history):
    blocks = []
    for item in history:
        blocks.append(f"{item.get('role')}: {item.get('text')}")
    return "\n".join(blocks)


def chat_reply(message, history):
    if client is None:
        return offline_reply(message)

    prompt = build_conversation(history[-8:])
    if prompt:
        prompt += "\n"
    prompt += "User: " + message
    prompt += "\nBot:"

    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                max_output_tokens=500,
            ),
        )
        return (response.text or "").strip() or offline_reply(message)
    except Exception:
        return offline_reply(message)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/sw.js")
def service_worker():
    return send_from_directory(app.static_folder, "sw.js")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    message = (data.get("message") or "").strip()
    if not message:
        return jsonify({"reply": "Please type a message."})
    history = data.get("history") or []
    return jsonify({"reply": chat_reply(message, history)})


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "5000")),
        debug=os.environ.get("FLASK_DEBUG", "1") == "1",
    )