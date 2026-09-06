import json
import math
import os
import re

from flask import Flask, jsonify, render_template, request, send_from_directory

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def read_api_key():
    env_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if env_key:
        return env_key
    key_file = os.path.join(BASE_DIR, "api_key.txt")
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
    "You are Shadow, the official AI assistant of Rayat Bahra Professional University "
    "(RBPU), Hoshiarpur campus (formerly Rayat Bahra Institute of Engineering and Nano "
    "Technology). Answer questions about admissions, courses, fees, scholarships, exams, "
    "campus facilities, sports, events and placements.\n\n"
    "FORMATTING RULES (very important):\n"
    "- Be thorough and detailed. Never give one-line answers, especially for fees, "
    "scholarships, admissions or courses. Expand with all figures and details you have.\n"
    "- Use bullet lists (each bullet starts with '- ') for categories, options and steps.\n"
    "- For numeric breakdowns (fee slabs, scholarship tiers, eligibility marks, courses "
    "lists), present a compact table using rows like '| Name | Amount/Details |'. No header "
    "row needed, just | cells | separated rows.\n"
    "- Do NOT mention images or say 'see the image'. The page images are shown automatically "
    "below the answer.\n"
    "- End your answer with the source page URL you used, as 'Source: rbpu.in/...'.\n\n"
    "When website content from rbpu.in is provided, answer ONLY from that content and "
    "include every relevant number found there, even if you must repeat them. If the answer "
    "is not in the provided content, briefly say it's handled by the university office and "
    "suggest the admission helpline +91 99884 00354.\n\n"
    "When no website content is provided, answer from general knowledge, stay friendly, "
    "and keep it reasonably short.\n\n"
    "Courses offered at the institute:\n"
    + "\n".join("- " + c for c in COURSES)
)

# Offline fallback answers used only when no API key is set or the API fails.
OFFLINE_ANSWERS = {
    "admission": "Admissions are open at Rayat Bahra Professional University, Hoshiarpur! Applications are accepted on rbpu.in / admissions.rbpu.in. Contact the admission helpline +91 99884 00354.",
    "course": "We offer B.Tech in Computer Science and Engineering, IT, Mechanical, Civil, as well as CSE specializations in AI & Machine Learning, Data Science, IoT, and Cyber Security.",
    "btech": "We offer B.Tech in Computer Science and Engineering, IT, Mechanical, Civil, as well as CSE specializations in AI & Machine Learning, Data Science, IoT, and Cyber Security.",
    "branch": "We offer B.Tech in Computer Science and Engineering, IT, Mechanical, Civil, as well as CSE specializations in AI & Machine Learning, Data Science, IoT, and Cyber Security.",
    "eligib": "Eligibility for UG degrees generally requires 10+2 from a recognized board with minimum aggregate marks as per the program. See the specific program page on rbpu.in.",
    "rbuset": "RBUSET is the Rayat Bahra University entrance test. Registration for RBUSET 2026 is open at rbpu.in/rbpuset-2026 and applications are submitted at admissions.rbpu.in.",
    "rbpuset": "RBUSET is the Rayat Bahra University entrance test. Registration for RBUSET 2026 is open at rbpu.in/rbpuset-2026 and applications are submitted at admissions.rbpu.in.",
    "exam": "RBPUSET is the university entrance exam. Check rbpu.in/rbpuset-2026 for registration and dates. Midterm exams use the timetable on the student portal.",
    "fee": "Fee details are published on rbpu.in under the program pages and the admission portal admissions.rbpu.in. Contact the accounts office for exact figures.",
    "scholarship": "RBPU offers merit and need-based scholarships with up to 100% tuition fee waiver, plus sports scholarships up to 60% fee waiver. Details are on rbpu.in.",
    "hostel": "Hostel rooms are available on a first-come basis. Apply through the hostel office at the Hoshiarpur campus. See rbpu.in for hostel capacity and facilities.",
    "library": "The library is open Monday-Saturday from 8 AM to 8 PM. A student ID is required to issue books.",
    "canteen": "The canteen is open from 8 AM to 6 PM and serves breakfast, lunch, and snacks.",
    "placement": "RBPU's placement cell has 1175+ recruiters who visited, 8100+ jobs offered, and the highest package is 1.25+ Crore. See rbpu.in/placements for details.",
    "sports": "RBPU has sports scholarships up to 60% fee waiver and an annual sports meet. Sports facilities are described on rbpu.in.",
    "event": "RBPU hosts fests and events like Fashionista and the Annual Fest. Check rbpu.in and news.rbpu.in for the latest events.",
    "fest": "RBPU hosts big annual fests and cultural events. The latest event updates are on news.rbpu.in.",
    "contact": "Admission helpline: +91 99884 00354. Hoshiarpur Campus, V.P.O Bohan, Tehsil & Distt Hoshiarpur, Punjab-146101.",
}

STOPWORDS = set(
    """
    a an the and or but if then else for from with without of in on at by to into
    what which who whom whose how why when where is are was were be been being
    do does did done have has had having can could will would shall should may
    might must not n't no yes please tell me my your our their this that these
    those about about between under over above through during before after again
    further once here there all any both each few more most other some such only
    own same so than too very just also its it's ok okay i you we they rbpu rayat bahra
    """.split()
)


def tokenize(text):
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) > 2 and t not in STOPWORDS]


class KB:
    def __init__(self, path):
        self.chunks = []
        self.doc_freq = {}
        self.doc_tokens = []
        self.num_docs = 0
        self.vectors = []
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            self.chunks = data.get("chunks", [])
            self.vectors = data.get("vectors", [])
            self.num_docs = len(self.chunks)
            for c in self.chunks:
                toks = tokenize(c.get("text", ""))
                self.doc_tokens.append(toks)
                for t in set(toks):
                    self.doc_freq[t] = self.doc_freq.get(t, 0) + 1
        except Exception:
            self.chunks = []

    def idf(self, term):
        n = self.num_docs
        df = self.doc_freq.get(term, 0)
        return math.log((1 + n) / (1 + df)) + 1 if n else 0

    def search(self, query, k=6, max_per_url=2):
        qtoks = tokenize(query)
        if not qtoks:
            return []
        scores = []
        for idx, toks in enumerate(self.doc_tokens):
            tf = {}
            for t in toks:
                tf[t] = tf.get(t, 0) + 1
            score = 0.0
            for t in set(qtoks):
                if t in tf:
                    score += (1 + math.log(tf[t])) * self.idf(t)
            if score > 0:
                scores.append((score, idx))
        scores.sort(key=lambda x: -x[0])
        picked, per_url = [], {}
        for score, idx in scores:
            url = self.chunks[idx].get("url", "")
            per_url[url] = per_url.get(url, 0) + 1
            if per_url[url] > max_per_url:
                continue
            picked.append((score, idx))
            if len(picked) >= k:
                break
        return picked

    def images_for(self, results, cap=4):
        seen, out = set(), []
        for score, idx in results:
            for u in self.chunks[idx].get("images", []):
                if u in seen:
                    continue
                seen.add(u)
                out.append(u)
                if len(out) >= cap:
                    return out
        return out


KB_PATH = os.path.join(BASE_DIR, "kb.json")
kb = KB(KB_PATH)


def build_context(results):
    lines = ["Relevant content from the official website rbpu.in:"]
    for score, idx in results:
        c = kb.chunks[idx]
        lines.append("\n--- " + c.get("url", "") + " | " + c.get("title", ""))
        lines.append(c.get("text", "")[:1600])
    return "\n".join(lines)


def offline_reply(message):
    text = message.lower()
    for keyword, reply in OFFLINE_ANSWERS.items():
        if re.search(rf"\b{re.escape(keyword)}s?\b", text):
            return reply, kb.images_for(kb.search(message, k=4, max_per_url=1))
    results = kb.search(message, k=3, max_per_url=2)
    if results:
        parts, srcs = [], set()
        for score, idx in results[:2]:
            chunk = kb.chunks[idx]
            parts.append(chunk.get("text", "")[:700])
            srcs.add(chunk.get("url", ""))
        body = "\n\n".join(parts)
        if srcs:
            body += "\n\nSource: " + ", ".join(s for s in srcs)
        return body, kb.images_for(results)
    return "I can help with admissions, fees, courses, scholarships, exams, hostel, campus life, sports and placements. You can also check rbpu.in.", []


def build_conversation(history):
    blocks = []
    for item in history:
        blocks.append(f"{item.get('role')}: {item.get('text')}")
    return "\n".join(blocks)


def chat_reply(message, history):
    if client is None:
        return offline_reply(message)

    results = kb.search(message, k=7, max_per_url=3)
    prompt = build_conversation(history[-8:])
    if prompt:
        prompt += "\n"
    if results:
        prompt += build_context(results) + "\n\n"
    prompt += "User: " + message
    prompt += "\nBot:"

    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                max_output_tokens=900,
            ),
        )
        text = (response.text or "").strip()
        if text:
            return text, kb.images_for(results)
        return offline_reply(message)
    except Exception:
        return offline_reply(message)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/sw.js")
def service_worker():
    return send_from_directory(app.static_folder, "sw.js")


CONTACT_INFO = {
    "name": "Rayat Bahra Professional University, Hoshiarpur Campus",
    "phones": ["+91 99884 00354", "+91 99884 01864"],
    "emails": ["admissionshspcampus@rayatbahra.com", "info@rayatbahra.com"],
    "address": "V.P.O Bohan, Tehsil & Distt Hoshiarpur, Punjab-146101",
}

HELPLINE_INFO = {
    "label": "Admission Helpline",
    "phones": ["+91 99884 00354", "+91 83606 28189"],
    "hours": "Monday to Saturday, 9 AM - 6 PM",
    "note": "For application support and admission counselling.",
}


@app.route("/api/contact")
def api_contact():
    return jsonify(CONTACT_INFO)


@app.route("/api/helpline")
def api_helpline():
    return jsonify(HELPLINE_INFO)


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    message = (data.get("message") or "").strip()
    if not message:
        return jsonify({"reply": "Please type a message.", "images": []})
    history = data.get("history") or []
    reply, images = chat_reply(message, history)
    return jsonify({"reply": reply, "images": images})


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "5000")),
        debug=os.environ.get("FLASK_DEBUG", "1") == "1",
    )