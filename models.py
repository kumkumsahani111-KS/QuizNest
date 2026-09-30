
from datetime import datetime, timezone

from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import (
    generate_password_hash,
    check_password_hash
)

db = SQLAlchemy()


# ---------------------------------
# TEACHER ACCOUNTS
# ---------------------------------

class Teacher(db.Model):
    __tablename__ = "teachers"

    id = db.Column(
        db.Integer,
        primary_key=True
    )

    name = db.Column(
        db.String(120),
        nullable=False
    )

    username = db.Column(
        db.String(80),
        unique=True,
        nullable=False,
        index=True
    )

    password_hash = db.Column(
        db.String(255),
        nullable=False
    )

    created_at = db.Column(
        db.DateTime,
        default=lambda: datetime.now(timezone.utc)
    )

    quizzes = db.relationship(
        "TeacherQuiz",
        back_populates="teacher",
        lazy=True
    )

    def set_password(self, password):
        self.password_hash = generate_password_hash(
            password
        )

    def check_password(self, password):
        return check_password_hash(
            self.password_hash,
            password
        )


# ---------------------------------
# TEACHER TESTS
# ---------------------------------

class TeacherQuiz(db.Model):
    __tablename__ = "teacher_quizzes"

    id = db.Column(
        db.Integer,
        primary_key=True
    )

    # Nullable temporarily to support
    # quizzes created before teacher accounts.
    teacher_id = db.Column(
        db.Integer,
        db.ForeignKey("teachers.id"),
        nullable=True,
        index=True
    )

    title = db.Column(
        db.String(200),
        nullable=False
    )

    status = db.Column(
        db.String(20),
        nullable=False,
        default="draft"
    )

    created_at = db.Column(
        db.DateTime,
        default=lambda: datetime.now(timezone.utc)
    )

    teacher = db.relationship(
        "Teacher",
        back_populates="quizzes"
    )

    questions = db.relationship(
        "TeacherQuestion",
        backref="quiz",
        cascade="all, delete-orphan",
        lazy=True,
        order_by="TeacherQuestion.position"
    )

    attempts = db.relationship(
        "TeacherAttempt",
        backref="quiz",
        cascade="all, delete-orphan",
        lazy=True
    )


# ---------------------------------
# TEACHER TEST QUESTIONS
# ---------------------------------

class TeacherQuestion(db.Model):
    __tablename__ = "teacher_questions"

    id = db.Column(
        db.Integer,
        primary_key=True
    )

    quiz_id = db.Column(
        db.Integer,
        db.ForeignKey("teacher_quizzes.id"),
        nullable=False
    )

    position = db.Column(
        db.Integer,
        nullable=False
    )

    question = db.Column(
        db.Text,
        nullable=False
    )

    option_a = db.Column(
        db.Text,
        nullable=False
    )

    option_b = db.Column(
        db.Text,
        nullable=False
    )

    option_c = db.Column(
        db.Text,
        nullable=False
    )

    option_d = db.Column(
        db.Text,
        nullable=False
    )

    correct_answer = db.Column(
        db.String(1),
        nullable=False
    )

    explanation = db.Column(
        db.Text,
        default=""
    )


# ---------------------------------
# STUDENT TEST RESULTS
# ---------------------------------

class TeacherAttempt(db.Model):
    __tablename__ = "teacher_attempts"

    id = db.Column(
        db.Integer,
        primary_key=True
    )

    quiz_id = db.Column(
        db.Integer,
        db.ForeignKey("teacher_quizzes.id"),
        nullable=False
    )

    student_name = db.Column(
        db.String(120),
        nullable=False
    )

    roll_no = db.Column(
        db.String(50),
        nullable=False
    )

    student_class = db.Column(
        db.String(100),
        nullable=False
    )

    score = db.Column(
        db.Integer,
        nullable=False
    )

    total = db.Column(
        db.Integer,
        nullable=False
    )

    percentage = db.Column(
        db.Float,
        nullable=False
    )

    submitted_at = db.Column(
        db.DateTime,
        default=lambda: datetime.now(timezone.utc)
    )

    # One attempt per student per test.
    __table_args__ = (
        db.UniqueConstraint(
            "quiz_id",
            "roll_no",
            "student_class",
            name="unique_student_test_attempt"
        ),
    )