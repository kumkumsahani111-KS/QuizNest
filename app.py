
import os
import io
import json
import re
from functools import wraps

from flask import (
    Flask, render_template, request, jsonify,
    session, redirect, url_for
)
from dotenv import load_dotenv
from google import genai
from google.genai import types
from groq import Groq
from pypdf import PdfReader
from PIL import Image, UnidentifiedImageError
from sqlalchemy.exc import IntegrityError

from models import (
    db, Teacher, TeacherQuiz,
    TeacherQuestion, TeacherAttempt
)


# ---------------------------------
# CONFIGURATION
# ---------------------------------

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY")

if not app.secret_key:
    raise RuntimeError(
        "SECRET_KEY is missing from your .env file."
    )

app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///quiznest.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024

db.init_app(app)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

GEMINI_MODEL = "gemini-3.5-flash-lite"
GROQ_MODEL = "openai/gpt-oss-120b"

ALLOWED_COUNTS = [5, 10, 15, 20]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
MAX_IMAGES = 10


# ---------------------------------
# STUDENT HOME AND SEPARATE PAGES
# ---------------------------------

@app.route("/")
def home():
    return render_template(
        "index.html",
        page="home"
    )


@app.route("/ask-ai")
def ask_ai_page():
    return render_template(
        "index.html",
        page="ask_ai"
    )


@app.route("/generate-quiz")
def generate_quiz_page():
    return render_template(
        "index.html",
        page="ai_quiz"
    )


@app.route("/study-quiz")
def study_quiz_page():
    return render_template(
        "index.html",
        page="material"
    )


# ---------------------------------
# GEMINI AI
# ---------------------------------

def ask_gemini(prompt, images=None):
    if not GEMINI_API_KEY:
        raise ValueError("Gemini API key is missing.")

    client = genai.Client(api_key=GEMINI_API_KEY)
    contents = [prompt]

    if images:
        for image_data, mime_type in images:
            contents.append(
                types.Part.from_bytes(
                    data=image_data,
                    mime_type=mime_type
                )
            )

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=contents
    )

    if not response.text:
        raise ValueError("Gemini returned an empty response.")

    return response.text


# ---------------------------------
# GROQ AI
# ---------------------------------

def ask_groq(prompt):
    if not GROQ_API_KEY:
        raise ValueError("Groq API key is missing.")

    client = Groq(api_key=GROQ_API_KEY)

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are QuizNest, a helpful AI study "
                    "assistant. Follow the requested output "
                    "format carefully."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.3
    )

    answer = response.choices[0].message.content

    if not answer:
        raise ValueError("Groq returned an empty response.")

    return answer


# ---------------------------------
# ASK AI API
# ---------------------------------

@app.route("/ask", methods=["POST"])
def ask_ai():
    data = request.get_json(silent=True) or {}
    question = data.get("question", "")

    if not isinstance(question, str):
        return jsonify({
            "error": "Please enter a valid question."
        }), 400

    question = question.strip()

    if not question:
        return jsonify({
            "error": "Please enter a question."
        }), 400

    if len(question) > 5000:
        return jsonify({
            "error": "Question is too long."
        }), 400

    prompt = (
        "You are QuizNest, a helpful AI study assistant. "
        "Explain concepts in simple English suitable for "
        "college students. Give examples when helpful.\n\n"
        f"Student question: {question}"
    )

    try:
        answer = ask_gemini(prompt)

        return jsonify({
            "answer": answer,
            "model": "Gemini"
        })

    except Exception as error:
        print(
            "Gemini request failed:",
            type(error).__name__
        )

    try:
        answer = ask_groq(prompt)

        return jsonify({
            "answer": answer,
            "model": "Groq"
        })

    except Exception as error:
        print(
            "Groq request failed:",
            type(error).__name__
        )

        return jsonify({
            "error": (
                "Both AI services are currently unavailable. "
                "Please try again later."
            )
        }), 503


# ---------------------------------
# QUIZ JSON VALIDATION
# ---------------------------------

def parse_quiz(ai_text, expected_count):
    cleaned = ai_text.strip()

    cleaned = re.sub(
        r"^```(?:json)?\s*",
        "",
        cleaned,
        flags=re.IGNORECASE
    )

    cleaned = re.sub(
        r"\s*```$",
        "",
        cleaned
    )

    try:
        quiz_data = json.loads(cleaned)

    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")

        if start == -1 or end == -1:
            raise ValueError("AI did not return valid JSON.")

        quiz_data = json.loads(cleaned[start:end + 1])

    if not isinstance(quiz_data, dict):
        raise ValueError("Invalid quiz structure.")

    questions = quiz_data.get("questions")

    if not isinstance(questions, list):
        raise ValueError("Quiz questions are missing.")

    validated_questions = []

    for item in questions:
        if not isinstance(item, dict):
            raise ValueError("Invalid question.")

        question = item.get("question")
        options = item.get("options")
        correct_answer = item.get("correct_answer")
        explanation = item.get("explanation", "")

        if (
            not isinstance(question, str)
            or not question.strip()
        ):
            raise ValueError("Invalid question text.")

        if (
            not isinstance(options, list)
            or len(options) != 4
        ):
            raise ValueError(
                "Each question must have four options."
            )

        if not all(
            isinstance(option, str) and option.strip()
            for option in options
        ):
            raise ValueError("Invalid option text.")

        if not isinstance(correct_answer, str):
            raise ValueError("Invalid correct answer.")

        correct_answer = correct_answer.strip().upper()

        if correct_answer not in ("A", "B", "C", "D"):
            raise ValueError("Invalid correct answer.")

        if not isinstance(explanation, str):
            explanation = ""

        validated_questions.append({
            "question": question.strip(),
            "options": [
                option.strip() for option in options
            ],
            "correct_answer": correct_answer,
            "explanation": explanation.strip()
        })

    if len(validated_questions) != expected_count:
        raise ValueError(
            "AI returned an incorrect number of questions."
        )

    return validated_questions


# ---------------------------------
# QUIZ PROMPT
# ---------------------------------

def create_quiz_prompt(
    topic,
    count,
    material=None,
    has_images=False
):
    if material is not None:
        source = (
            "Create questions using ONLY the provided "
            "study material. Do not invent unsupported facts.\n\n"
            f"Written study material:\n{material}"
        )

        if has_images:
            source += (
                "\n\nAlso carefully read the attached images. "
                "Use their readable study content to create "
                "questions. Ignore decorative elements."
            )
    else:
        source = f"Quiz topic: {topic}"

    return f"""
You are QuizNest, an AI quiz generator.

{source}

Create exactly {count} multiple-choice questions.

Requirements:
- Use simple English suitable for college students.
- Each question must have exactly four options.
- Each question must have one correct answer.
- Include a short explanation for every answer.
- Avoid duplicate questions.
- Return ONLY valid JSON.
- Do not include Markdown or extra text.

Use this JSON structure:

{{
  "questions": [
    {{
      "question": "Example question?",
      "options": [
        "First option",
        "Second option",
        "Third option",
        "Fourth option"
      ],
      "correct_answer": "A",
      "explanation": "Short explanation."
    }}
  ]
}}

The correct_answer must be A, B, C or D.

Generate exactly {count} questions.
"""


# ---------------------------------
# COMMON QUIZ GENERATOR
# ---------------------------------

def generate_questions(prompt, count, images=None):
    try:
        ai_text = ask_gemini(prompt, images=images)
        questions = parse_quiz(ai_text, count)

        return {
            "questions": questions,
            "model": "Gemini"
        }

    except Exception as error:
        print(
            "Gemini quiz failed:",
            type(error).__name__
        )

    # Groq cannot read uploaded images here.
    if images:
        return None

    try:
        ai_text = ask_groq(prompt)
        questions = parse_quiz(ai_text, count)

        return {
            "questions": questions,
            "model": "Groq"
        }

    except Exception as error:
        print(
            "Groq quiz failed:",
            type(error).__name__
        )

    return None


# ---------------------------------
# NORMAL AI QUIZ API
# ---------------------------------

@app.route("/quiz", methods=["POST"])
def generate_quiz():
    data = request.get_json(silent=True) or {}

    topic = data.get("topic", "")
    count = data.get("count", 5)

    if not isinstance(topic, str):
        return jsonify({
            "error": "Please enter a valid topic."
        }), 400

    topic = topic.strip()

    if not topic:
        return jsonify({
            "error": "Please enter a quiz topic."
        }), 400

    if len(topic) > 200:
        return jsonify({
            "error": "Topic is too long."
        }), 400

    try:
        count = int(count)

    except (ValueError, TypeError):
        return jsonify({
            "error": "Invalid number of questions."
        }), 400

    if count not in ALLOWED_COUNTS:
        return jsonify({
            "error": "Choose 5, 10, 15 or 20 questions."
        }), 400

    prompt = create_quiz_prompt(topic, count)
    result = generate_questions(prompt, count)

    if result is None:
        return jsonify({
            "error": (
                "Unable to generate the quiz right now. "
                "Please try again."
            )
        }), 503

    return jsonify({
        "topic": topic,
        "questions": result["questions"],
        "model": result["model"]
    })


# ---------------------------------
# READ PDF AND TXT
# ---------------------------------

def extract_file_text(uploaded_file):
    filename = uploaded_file.filename or ""
    extension = os.path.splitext(filename.lower())[1]

    if extension == ".txt":
        raw_data = uploaded_file.read()

        try:
            return raw_data.decode("utf-8-sig")

        except UnicodeDecodeError:
            raise ValueError(
                "Please upload a UTF-8 TXT file."
            )

    if extension == ".pdf":
        try:
            pdf_data = uploaded_file.read()
            reader = PdfReader(io.BytesIO(pdf_data))

            if reader.is_encrypted:
                raise ValueError(
                    "Password-protected PDFs are not supported."
                )

            pages = []

            for page in reader.pages:
                page_text = page.extract_text() or ""

                if page_text.strip():
                    pages.append(page_text)

            return "\n\n".join(pages)

        except ValueError:
            raise

        except Exception:
            raise ValueError(
                "Unable to read this PDF. "
                "Please upload a valid PDF."
            )

    raise ValueError(
        "Only PDF, TXT, JPG, JPEG, PNG and WEBP "
        "files are supported."
    )


# ---------------------------------
# READ AND VALIDATE IMAGE
# ---------------------------------

def extract_image(uploaded_file):
    filename = uploaded_file.filename or ""
    extension = os.path.splitext(filename.lower())[1]

    if extension not in IMAGE_EXTENSIONS:
        raise ValueError("Unsupported image format.")

    image_data = uploaded_file.read()

    if not image_data:
        raise ValueError("The uploaded image is empty.")

    try:
        with Image.open(io.BytesIO(image_data)) as image:
            image.verify()

        with Image.open(io.BytesIO(image_data)) as image:
            width, height = image.size

            if width * height > 20_000_000:
                raise ValueError(
                    "Image is too large. "
                    "Please use a smaller image."
                )

            actual_format = image.format

    except (
        UnidentifiedImageError,
        OSError,
        Image.DecompressionBombError
    ):
        raise ValueError(
            "Unable to read this image. "
            "Please upload a valid image."
        )

    supported_formats = {
        "JPEG": "image/jpeg",
        "PNG": "image/png",
        "WEBP": "image/webp"
    }

    mime_type = supported_formats.get(actual_format)

    if not mime_type:
        raise ValueError(
            "Only JPG, JPEG, PNG and WEBP "
            "images are supported."
        )

    return image_data, mime_type


# ---------------------------------
# STUDY MATERIAL QUIZ API
# ---------------------------------

@app.route("/material-quiz", methods=["POST"])
def material_quiz():
    study_text = request.form.get("study_text", "")
    count = request.form.get("count", "5")

    try:
        count = int(count)

    except (ValueError, TypeError):
        return jsonify({
            "error": "Invalid question count."
        }), 400

    if count not in ALLOWED_COUNTS:
        return jsonify({
            "error": "Choose 5, 10, 15 or 20 questions."
        }), 400

    uploaded_files = request.files.getlist("files")
    all_text = []
    images = []

    if study_text.strip():
        all_text.append(study_text.strip())

    for uploaded_file in uploaded_files:
        if not uploaded_file.filename:
            continue

        extension = os.path.splitext(
            uploaded_file.filename.lower()
        )[1]

        try:
            if extension in IMAGE_EXTENSIONS:
                if len(images) >= MAX_IMAGES:
                    return jsonify({
                        "error": (
                            "You can upload a maximum "
                            "of 10 images."
                        )
                    }), 400

                image_data, mime_type = extract_image(
                    uploaded_file
                )

                images.append((image_data, mime_type))

            elif extension in (".pdf", ".txt"):
                extracted_text = extract_file_text(
                    uploaded_file
                )

                if extracted_text.strip():
                    all_text.append(extracted_text.strip())

            else:
                return jsonify({
                    "error": (
                        "Unsupported file. Use PDF, TXT, "
                        "JPG, JPEG, PNG or WEBP."
                    )
                }), 400

        except ValueError as error:
            return jsonify({
                "error": str(error)
            }), 400

    if not all_text and not images:
        return jsonify({
            "error": (
                "Please upload study material "
                "or paste your notes."
            )
        }), 400

    combined_text = "\n\n".join(all_text)

    if not images and len(combined_text) < 50:
        return jsonify({
            "error": (
                "Study material is too short. "
                "Please provide more information."
            )
        }), 400

    if len(combined_text) > 30000:
        return jsonify({
            "error": (
                "Study material is too long. "
                "Please use shorter notes."
            )
        }), 400

    prompt = create_quiz_prompt(
        topic="Study Material",
        count=count,
        material=combined_text,
        has_images=bool(images)
    )

    result = generate_questions(
        prompt,
        count,
        images=images
    )

    if result is None:
        return jsonify({
            "error": (
                "Unable to generate a quiz from this "
                "material right now. Please try again."
            )
        }), 503

    return jsonify({
        "topic": "Study Material",
        "questions": result["questions"],
        "model": result["model"]
    })


# ---------------------------------
# TEACHER AUTHENTICATION
# ---------------------------------

def get_current_teacher():
    teacher_id = session.get("teacher_id")

    if not isinstance(teacher_id, int):
        return None

    return db.session.get(Teacher, teacher_id)


def teacher_required(function):
    @wraps(function)
    def wrapper(*args, **kwargs):
        teacher = get_current_teacher()

        if teacher is None:
            session.clear()

            is_api_request = (
                request.path.startswith("/teacher/generate")
                or request.path.startswith("/teacher/quizzes")
                or request.path == "/teacher/results"
                or request.is_json
            )

            if is_api_request:
                return jsonify({
                    "error": "Please log in as a teacher."
                }), 401

            return redirect(url_for("teacher_login"))

        return function(*args, **kwargs)

    return wrapper


# ---------------------------------
# TEACHER REGISTRATION
# ---------------------------------

@app.route("/teacher/register", methods=["GET", "POST"])
def teacher_register():
    if get_current_teacher():
        return redirect(url_for("teacher_dashboard"))

    if request.method == "GET":
        return render_template(
            "teacher_register.html",
            error=None
        )

    name = request.form.get("name", "").strip()

    username = request.form.get(
        "username", ""
    ).strip().lower()

    password = request.form.get("password", "")

    confirm_password = request.form.get(
        "confirm_password", ""
    )

    error = None

    if not name or len(name) > 120:
        error = "Please enter a valid name."

    elif not re.fullmatch(r"[a-z0-9_]{3,30}", username):
        error = (
            "Username must contain 3 to 30 lowercase "
            "letters, numbers or underscores."
        )

    elif len(password) < 8:
        error = (
            "Password must contain at least 8 characters."
        )

    elif len(password) > 128:
        error = "Password is too long."

    elif password != confirm_password:
        error = "Passwords do not match."

    elif Teacher.query.filter_by(
        username=username
    ).first():
        error = "This username is already taken."

    if error:
        return render_template(
            "teacher_register.html",
            error=error
        ), 400

    teacher = Teacher(
        name=name,
        username=username
    )

    teacher.set_password(password)
    db.session.add(teacher)

    try:
        db.session.commit()

    except IntegrityError:
        db.session.rollback()

        return render_template(
            "teacher_register.html",
            error="This username is already taken."
        ), 409

    session.clear()
    session["teacher_id"] = teacher.id

    return redirect(url_for("teacher_dashboard"))


# ---------------------------------
# TEACHER LOGIN
# ---------------------------------

@app.route("/teacher/login", methods=["GET", "POST"])
def teacher_login():
    if get_current_teacher():
        return redirect(url_for("teacher_dashboard"))

    if request.method == "POST":
        username = request.form.get(
            "username", ""
        ).strip().lower()

        password = request.form.get("password", "")

        teacher = Teacher.query.filter_by(
            username=username
        ).first()

        if teacher and teacher.check_password(password):
            session.clear()
            session["teacher_id"] = teacher.id

            return redirect(url_for("teacher_dashboard"))

        return render_template(
            "teacher.html",
            page="login",
            error="Invalid username or password."
        ), 401

    return render_template(
        "teacher.html",
        page="login",
        error=None
    )


# ---------------------------------
# TEACHER DASHBOARD
# ---------------------------------

@app.route("/teacher")
@teacher_required
def teacher_dashboard():
    teacher = get_current_teacher()

    quizzes = TeacherQuiz.query.filter_by(
        teacher_id=teacher.id
    ).order_by(
        TeacherQuiz.created_at.desc()
    ).all()

    return render_template(
        "teacher.html",
        page="dashboard",
        teacher=teacher,
        quizzes=quizzes
    )


# ---------------------------------
# NEW: SEPARATE TEACHER PAGES
# ---------------------------------

@app.route("/teacher/create")
@teacher_required
def teacher_create_page():
    return render_template(
        "teacher.html",
        page="create",
        teacher=get_current_teacher()
    )


@app.route("/teacher/my-tests")
@teacher_required
def teacher_my_tests_page():
    return render_template(
        "teacher.html",
        page="my_tests",
        teacher=get_current_teacher()
    )


@app.route("/teacher/student-results")
@teacher_required
def teacher_student_results_page():
    return render_template(
        "teacher.html",
        page="results",
        teacher=get_current_teacher()
    )


# ---------------------------------
# TEACHER LOGOUT
# ---------------------------------

@app.route("/teacher/logout", methods=["POST"])
@teacher_required
def teacher_logout():
    session.clear()

    return redirect(url_for("teacher_login"))


# ---------------------------------
# VALIDATE TEACHER QUESTIONS
# ---------------------------------

def validate_teacher_questions(questions):
    if not isinstance(questions, list):
        raise ValueError("Questions must be a list.")

    if not 1 <= len(questions) <= 30:
        raise ValueError(
            "A test must have 1 to 30 questions."
        )

    validated = []

    for item in questions:
        if not isinstance(item, dict):
            raise ValueError("Invalid question.")

        question = item.get("question", "")
        options = item.get("options", [])
        correct = item.get("correct_answer", "")
        explanation = item.get("explanation", "")

        if (
            not isinstance(question, str)
            or not question.strip()
        ):
            raise ValueError(
                "Every question needs question text."
            )

        if (
            not isinstance(options, list)
            or len(options) != 4
            or not all(
                isinstance(option, str) and option.strip()
                for option in options
            )
        ):
            raise ValueError(
                "Every question must have four non-empty options."
            )

        if not isinstance(correct, str):
            raise ValueError("Invalid correct answer.")

        correct = correct.strip().upper()

        if correct not in ("A", "B", "C", "D"):
            raise ValueError(
                "Correct answer must be A, B, C or D."
            )

        if not isinstance(explanation, str):
            explanation = ""

        validated.append({
            "question": question.strip(),
            "options": [
                option.strip() for option in options
            ],
            "correct_answer": correct,
            "explanation": explanation.strip()
        })

    return validated


# ---------------------------------
# CONVERT SAVED TEST TO JSON
# ---------------------------------

def teacher_quiz_to_dict(quiz):
    return {
        "id": quiz.id,
        "title": quiz.title,
        "status": quiz.status,
        "question_count": len(quiz.questions),
        "attempt_count": len(quiz.attempts),
        "questions": [
            {
                "id": item.id,
                "question": item.question,
                "options": [
                    item.option_a,
                    item.option_b,
                    item.option_c,
                    item.option_d
                ],
                "correct_answer": item.correct_answer,
                "explanation": item.explanation
            }
            for item in quiz.questions
        ]
    }


# ---------------------------------
# SAVE QUESTION RECORDS
# ---------------------------------

def replace_teacher_questions(quiz, questions):
    quiz.questions.clear()
    db.session.flush()

    for position, item in enumerate(questions, start=1):
        quiz.questions.append(
            TeacherQuestion(
                position=position,
                question=item["question"],
                option_a=item["options"][0],
                option_b=item["options"][1],
                option_c=item["options"][2],
                option_d=item["options"][3],
                correct_answer=item["correct_answer"],
                explanation=item["explanation"]
            )
        )


# ---------------------------------
# GENERATE TEACHER MCQS
# ---------------------------------

@app.route("/teacher/generate", methods=["POST"])
@teacher_required
def teacher_generate():
    title = request.form.get("title", "").strip()
    study_text = request.form.get("study_text", "").strip()

    try:
        count = int(request.form.get("count", "10"))

    except (ValueError, TypeError):
        return jsonify({
            "error": "Enter a valid question count."
        }), 400

    if not title or len(title) > 200:
        return jsonify({
            "error": (
                "Enter a test title "
                "(maximum 200 characters)."
            )
        }), 400

    if not 1 <= count <= 30:
        return jsonify({
            "error": "Choose between 1 and 30 questions."
        }), 400

    all_text = [study_text] if study_text else []
    images = []

    for uploaded_file in request.files.getlist("files"):
        if not uploaded_file.filename:
            continue

        extension = os.path.splitext(
            uploaded_file.filename.lower()
        )[1]

        try:
            if extension in IMAGE_EXTENSIONS:
                if len(images) >= MAX_IMAGES:
                    raise ValueError(
                        "You can upload a maximum of 10 images."
                    )

                images.append(
                    extract_image(uploaded_file)
                )

            elif extension in (".pdf", ".txt"):
                extracted = extract_file_text(uploaded_file)

                if extracted.strip():
                    all_text.append(extracted.strip())

            else:
                raise ValueError(
                    "Only PDF, TXT, JPG, JPEG, PNG and WEBP "
                    "files are supported."
                )

        except ValueError as error:
            return jsonify({
                "error": str(error)
            }), 400

    combined_text = "\n\n".join(all_text)

    if not combined_text and not images:
        return jsonify({
            "error": (
                "Upload study material or paste your notes."
            )
        }), 400

    if not images and len(combined_text) < 50:
        return jsonify({
            "error": (
                "Please provide at least 50 characters "
                "of notes."
            )
        }), 400

    if len(combined_text) > 30000:
        return jsonify({
            "error": (
                "Study material is too long "
                "(maximum 30,000 characters)."
            )
        }), 400

    prompt = create_quiz_prompt(
        topic=title,
        count=count,
        material=combined_text,
        has_images=bool(images)
    )

    result = generate_questions(
        prompt,
        count,
        images=images
    )

    if result is None:
        return jsonify({
            "error": (
                "AI could not generate this test right now. "
                "Please try again."
            )
        }), 503

    return jsonify({
        "title": title,
        "questions": result["questions"],
        "model": result["model"]
    })


# ---------------------------------
# LIST AND SAVE TEACHER TESTS
# ---------------------------------

@app.route("/teacher/quizzes", methods=["GET", "POST"])
@teacher_required
def teacher_quizzes():
    teacher = get_current_teacher()

    if request.method == "GET":
        quizzes = TeacherQuiz.query.filter_by(
            teacher_id=teacher.id
        ).order_by(
            TeacherQuiz.created_at.desc()
        ).all()

        return jsonify({
            "quizzes": [
                {
                    "id": quiz.id,
                    "title": quiz.title,
                    "status": quiz.status,
                    "question_count": len(quiz.questions),
                    "attempt_count": len(quiz.attempts)
                }
                for quiz in quizzes
            ]
        })

    data = request.get_json(silent=True) or {}

    title = data.get("title", "")
    status = data.get("status", "draft")

    if (
        not isinstance(title, str)
        or not title.strip()
    ):
        return jsonify({
            "error": "Test title is required."
        }), 400

    title = title.strip()

    if len(title) > 200:
        return jsonify({
            "error": "Test title is too long."
        }), 400

    if status not in ("draft", "published"):
        return jsonify({
            "error": "Invalid test status."
        }), 400

    try:
        questions = validate_teacher_questions(
            data.get("questions")
        )

    except ValueError as error:
        return jsonify({
            "error": str(error)
        }), 400

    quiz = TeacherQuiz(
        teacher_id=teacher.id,
        title=title,
        status=status
    )

    db.session.add(quiz)

    replace_teacher_questions(
        quiz,
        questions
    )

    db.session.commit()

    return jsonify({
        "message": "Test saved successfully.",
        "quiz": teacher_quiz_to_dict(quiz)
    }), 201


# ---------------------------------
# VIEW, EDIT AND DELETE TEACHER TEST
# ---------------------------------

@app.route(
    "/teacher/quizzes/<int:quiz_id>",
    methods=["GET", "PUT", "DELETE"]
)
@teacher_required
def teacher_quiz_detail(quiz_id):
    teacher = get_current_teacher()

    quiz = TeacherQuiz.query.filter_by(
        id=quiz_id,
        teacher_id=teacher.id
    ).first_or_404()

    if request.method == "GET":
        return jsonify({
            "quiz": teacher_quiz_to_dict(quiz)
        })

    # Protect existing student results.
    if quiz.attempts:
        return jsonify({
            "error": (
                "This test already has student attempts. "
                "It cannot be edited or deleted."
            )
        }), 409

    if request.method == "DELETE":
        db.session.delete(quiz)
        db.session.commit()

        return jsonify({
            "message": "Test deleted successfully."
        })

    data = request.get_json(silent=True) or {}

    title = data.get("title", quiz.title)
    status = data.get("status", quiz.status)

    if (
        not isinstance(title, str)
        or not title.strip()
    ):
        return jsonify({
            "error": "Test title is required."
        }), 400

    title = title.strip()

    if len(title) > 200:
        return jsonify({
            "error": "Test title is too long."
        }), 400

    if status not in ("draft", "published"):
        return jsonify({
            "error": "Invalid test status."
        }), 400

    try:
        questions = validate_teacher_questions(
            data.get("questions")
        )

    except ValueError as error:
        return jsonify({
            "error": str(error)
        }), 400

    quiz.title = title
    quiz.status = status

    replace_teacher_questions(
        quiz,
        questions
    )

    db.session.commit()

    return jsonify({
        "message": "Test updated successfully.",
        "quiz": teacher_quiz_to_dict(quiz)
    })


# ---------------------------------
# PUBLISH TEACHER TEST
# ---------------------------------

@app.route(
    "/teacher/quizzes/<int:quiz_id>/publish",
    methods=["POST"]
)
@teacher_required
def publish_teacher_quiz(quiz_id):
    teacher = get_current_teacher()

    quiz = TeacherQuiz.query.filter_by(
        id=quiz_id,
        teacher_id=teacher.id
    ).first_or_404()

    if not quiz.questions:
        return jsonify({
            "error": "Add questions before publishing."
        }), 400

    quiz.status = "published"
    db.session.commit()

    return jsonify({
        "message": "Test published successfully.",
        "quiz_id": quiz.id
    })


# ---------------------------------
# TEACHER RESULTS API
# ---------------------------------

# This route returns JSON to the Results page.
# The visible HTML page uses /teacher/student-results.

@app.route("/teacher/results", methods=["GET"])
@teacher_required
def teacher_results():
    teacher = get_current_teacher()

    attempts = TeacherAttempt.query.join(
        TeacherQuiz
    ).filter(
        TeacherQuiz.teacher_id == teacher.id
    ).order_by(
        TeacherAttempt.submitted_at.desc()
    ).all()

    return jsonify({
        "results": [
            {
                "id": attempt.id,
                "test_id": attempt.quiz_id,
                "test_title": attempt.quiz.title,
                "student_name": attempt.student_name,
                "roll_no": attempt.roll_no,
                "student_class": attempt.student_class,
                "score": attempt.score,
                "total": attempt.total,
                "percentage": attempt.percentage
            }
            for attempt in attempts
        ]
    })


# ---------------------------------
# STUDENT TEST PORTAL
# ---------------------------------

@app.route("/student-tests")
def student_tests():
    # Only published tests are visible to students.
    quizzes = TeacherQuiz.query.filter_by(
        status="published"
    ).order_by(
        TeacherQuiz.created_at.desc()
    ).all()

    return render_template(
        "student_tests.html",
        page="list",
        quizzes=quizzes
    )


# ---------------------------------
# STUDENT OPENS A PUBLISHED TEST
# ---------------------------------

@app.route("/student-tests/<int:quiz_id>")
def student_test_detail(quiz_id):
    quiz = TeacherQuiz.query.filter_by(
        id=quiz_id,
        status="published"
    ).first_or_404()

    if not quiz.questions:
        return redirect(
            url_for("student_tests")
        )

    # Never expose correct answers before submission.
    questions = [
        {
            "id": question.id,
            "question": question.question,
            "options": [
                question.option_a,
                question.option_b,
                question.option_c,
                question.option_d
            ]
        }
        for question in quiz.questions
    ]

    return render_template(
        "student_tests.html",
        page="attempt",
        quiz=quiz,
        questions=questions
    )


# ---------------------------------
# STUDENT SUBMITS A TEST
# ---------------------------------

@app.route(
    "/student-tests/<int:quiz_id>/submit",
    methods=["POST"]
)
def submit_student_test(quiz_id):
    quiz = TeacherQuiz.query.filter_by(
        id=quiz_id,
        status="published"
    ).first_or_404()

    if not quiz.questions:
        return jsonify({
            "error": "This test has no questions."
        }), 400

    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({
            "error": "Invalid submission."
        }), 400

    student_name = data.get("student_name", "")
    roll_no = data.get("roll_no", "")
    student_class = data.get("student_class", "")
    answers = data.get("answers", {})

    if (
        not isinstance(student_name, str)
        or not student_name.strip()
        or len(student_name.strip()) > 120
    ):
        return jsonify({
            "error": "Enter a valid student name."
        }), 400

    if (
        not isinstance(roll_no, str)
        or not roll_no.strip()
        or len(roll_no.strip()) > 50
    ):
        return jsonify({
            "error": "Enter a valid roll number."
        }), 400

    if (
        not isinstance(student_class, str)
        or not student_class.strip()
        or len(student_class.strip()) > 100
    ):
        return jsonify({
            "error": "Enter a valid class."
        }), 400

    if not isinstance(answers, dict):
        return jsonify({
            "error": "Invalid answers."
        }), 400

    student_name = student_name.strip()
    roll_no = roll_no.strip()
    student_class = student_class.strip()

    # Only one submission per roll number and class.
    previous_attempt = TeacherAttempt.query.filter_by(
        quiz_id=quiz.id,
        roll_no=roll_no,
        student_class=student_class
    ).first()

    if previous_attempt:
        return jsonify({
            "error": (
                "An attempt already exists for this "
                "roll number and class."
            )
        }), 409

    valid_question_ids = {
        str(question.id)
        for question in quiz.questions
    }

    # Reject answers for questions outside this test.
    if not all(
        isinstance(key, str)
        and key in valid_question_ids
        for key in answers
    ):
        return jsonify({
            "error": "Invalid question submitted."
        }), 400

    score = 0
    total = len(quiz.questions)

    for question in quiz.questions:
        answer = answers.get(
            str(question.id),
            ""
        )

        if not isinstance(answer, str):
            return jsonify({
                "error": "Invalid answer format."
            }), 400

        answer = answer.strip().upper()

        if answer and answer not in (
            "A", "B", "C", "D"
        ):
            return jsonify({
                "error": "Invalid answer option."
            }), 400

        if answer == question.correct_answer:
            score += 1

    percentage = round(
        (score / total) * 100,
        2
    )

    attempt = TeacherAttempt(
        quiz_id=quiz.id,
        student_name=student_name,
        roll_no=roll_no,
        student_class=student_class,
        score=score,
        total=total,
        percentage=percentage
    )

    db.session.add(attempt)

    try:
        db.session.commit()

    except IntegrityError:
        db.session.rollback()

        return jsonify({
            "error": (
                "An attempt already exists for this "
                "roll number and class."
            )
        }), 409

    return jsonify({
        "message": "Test submitted successfully!",
        "test_title": quiz.title,
        "student_name": student_name,
        "score": score,
        "total": total,
        "percentage": percentage
    }), 201


# ---------------------------------
# FILE SIZE ERROR
# ---------------------------------

@app.errorhandler(413)
def file_too_large(error):
    return jsonify({
        "error": (
            "Files are too large. "
            "Maximum total upload size is 25 MB."
        )
    }), 413


# ---------------------------------
# CREATE DATABASE AND START SERVER
# ---------------------------------

with app.app_context():
    db.create_all()
    print("QuizNest database is ready!")


if __name__ == "__main__":
    app.run(debug=True)