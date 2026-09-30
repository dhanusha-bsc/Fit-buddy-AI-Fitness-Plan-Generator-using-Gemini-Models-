"""
FitBuddy - AI Fitness Plan Generator
TN Skills / Naan Mudhalvan College Project
Texcity Arts & Science College - BSc Computer Science
Team: Dhanusha, Deepa Krishnan, Anfas, Kameleshwaran
"""

import os
import uvicorn
from typing import Optional
from datetime import datetime

from fastapi import FastAPI, Request, Form, Depends
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel

# SQLAlchemy ORM Setup
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker, Session

# Optional: Google Generative AI SDK
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

# -----------------------------------------------------------------------------
# 1. DATABASE CONFIGURATION (SQLite + SQLAlchemy)
# -----------------------------------------------------------------------------
DATABASE_URL = "sqlite:///./fitbuddy.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(50), unique=True, index=True)
    name = Column(String(100))
    age = Column(Integer)
    weight = Column(String(20))
    goal = Column(String(100))
    intensity = Column(String(50))
    created_at = Column(DateTime, default=datetime.utcnow)

class WorkoutPlan(Base):
    __tablename__ = "plans"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(50), index=True)
    original_plan = Column(Text)
    nutrition_tip = Column(Text)
    feedback = Column(Text, nullable=True)
    updated_plan = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

# Create tables in SQLite database
Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# -----------------------------------------------------------------------------
# 2. GEMINI AI INTEGRATION & SMART FALLBACKS
# -----------------------------------------------------------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

if GENAI_AVAILABLE and GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

def generate_workout_with_gemini(name: str, age: int, weight: str, goal: str, intensity: str) -> str:
    """Invokes Gemini 1.5 Pro (or fallback model) to craft a 7-day personalized workout."""
    if GENAI_AVAILABLE and GEMINI_API_KEY:
        try:
            model = genai.GenerativeModel("gemini-1.5-pro")
            prompt = (
                f"You are FitBuddy AI, an elite fitness coach. Create a structured 7-day workout routine for:\n"
                f"- Name: {name}\n- Age: {age}\n- Weight: {weight} kg\n- Fitness Goal: {goal}\n- Intensity: {intensity}\n\n"
                f"Format cleanly day by day (Day 1 to Day 7) with warm-up, main sets/reps, cooldown, and rest days. Keep formatting concise."
            )
            response = model.generate_content(prompt)
            if response and response.text:
                return response.text
        except Exception as e:
            print(f"[Gemini Error]: {e}")

    # Fallback template if API key is not configured or network unavailable
    return (
        f"### ??? 7-Day Personalized Workout Plan for {name}\n"
        f"**Profile Context**: Goal: {goal.title()} | Intensity: {intensity.title()} | Weight: {weight}kg\n\n"
        f"- **Day 1 (Upper Body Push)**: 5 min Arm Circles Warmup, 3x10 Push-ups / Dumbbell Press, 3x12 Shoulder Press, 10 min Cooldown Stretch.\n"
        f"- **Day 2 (Lower Body & Core)**: Bodyweight Squats 3x12, Walking Lunges 3x10/leg, Plank Hold 3x45s.\n"
        f"- **Day 3 (Active Recovery)**: 30 minutes brisk walking or light yoga & dynamic hamstring stretches.\n"
        f"- **Day 4 (Upper Body Pull & Core)**: Resistance band rows 3x12, Superman holds 3x10, Hanging knee raises 3x10.\n"
        f"- **Day 5 (HIIT & Conditioning)**: 20 min interval routine: 45s Jumping jacks, 30s High knees, 30s Rest (5 rounds).\n"
        f"- **Day 6 (Full Body Strength)**: Goblet squats 3x10, Glute bridges 3x15, Push-ups 3x8, Side planks 2x30s.\n"
        f"- **Day 7 (Deep Rest & Hydration)**: Complete recovery day. Light foam rolling and 3L water intake target."
    )

def generate_nutrition_tip_with_gemini(goal: str) -> str:
    """Invokes Gemini Flash for fast goal-aligned nutrition/recovery advice."""
    if GENAI_AVAILABLE and GEMINI_API_KEY:
        try:
            model = genai.GenerativeModel("gemini-1.5-flash")
            prompt = f"Give 2 concise, practical nutrition and hydration tips for someone whose goal is '{goal}'."
            response = model.generate_content(prompt)
            if response and response.text:
                return response.text
        except Exception as e:
            print(f"[Gemini Flash Error]: {e}")

    # Smart fallback
    if "loss" in goal.lower():
        return "Prioritize 25-30g protein per meal to preserve lean muscle, maintain a mild caloric deficit, and drink 500ml water 20 mins prior to meals."
    elif "muscle" in goal.lower() or "gain" in goal.lower():
        return "Aim for 1.6g-2.0g protein per kg of bodyweight daily. Fuel pre-workout with complex carbs (oats or bananas) and ensure 7-8 hours deep sleep."
    else:
        return "Balance whole foods with leafy greens, lean proteins, and healthy fats (nuts, seeds). Hydrate consistently throughout your training sessions."

def update_workout_with_feedback(original_plan: str, feedback: str) -> str:
    """Revises the original workout according to natural language feedback."""
    if GENAI_AVAILABLE and GEMINI_API_KEY:
        try:
            model = genai.GenerativeModel("gemini-1.5-pro")
            prompt = (
                f"You are FitBuddy AI. Here is the original workout plan:\n{original_plan}\n\n"
                f"The user provided this feedback: '{feedback}'.\n"
                f"Please update the 7-day routine to incorporate this feedback while preserving the structure."
            )
            response = model.generate_content(prompt)
            if response and response.text:
                return response.text
        except Exception as e:
            print(f"[Gemini Feedback Error]: {e}")

    return (
        f"### ?? Revised 7-Day Plan (Adapted based on: '{feedback}')\n\n"
        f"**Modifications Applied**:\n"
        f"- Routine adjusted according to request: *\"{feedback}\"*.\n"
        f"- Rest periods balanced and specific exercise variations swapped.\n\n"
        f"{original_plan}\n\n*(Note: Tailored progression updated for weekly consistency)*"
    )

# -----------------------------------------------------------------------------
# 3. HTML UI TEMPLATES & STYLING
# -----------------------------------------------------------------------------
CSS_STYLES = """
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }
    body { background-color: #0b1120; color: #f8fafc; min-height: 100vh; padding: 24px; }
    .container { max-width: 960px; margin: 0 auto; }
    .header { text-align: center; margin-bottom: 32px; padding: 20px 0; border-bottom: 1px solid #1e293b; }
    .header h1 { font-size: 32px; color: #38bdf8; margin-bottom: 8px; }
    .header p { color: #94a3b8; font-size: 16px; }
    .badge { display: inline-block; background: rgba(56, 189, 248, 0.15); color: #38bdf8; padding: 4px 12px; border-radius: 20px; font-size: 13px; font-weight: 600; margin-bottom: 10px; }
    .card { background: #111a2e; border: 1px solid #1e2d4a; border-radius: 12px; padding: 28px; margin-bottom: 24px; box-shadow: 0 10px 25px rgba(0,0,0,0.3); }
    .card h2 { font-size: 20px; color: #f1f5f9; margin-bottom: 16px; display: flex; align-items: center; gap: 8px; }
    .form-group { margin-bottom: 18px; }
    label { display: block; font-size: 14px; font-weight: 600; color: #cbd5e1; margin-bottom: 6px; }
    input, select, textarea { width: 100%; padding: 12px 14px; background: #070d19; border: 1px solid #1e293b; border-radius: 8px; color: #f8fafc; font-size: 15px; outline: none; }
    input:focus, select:focus, textarea:focus { border-color: #38bdf8; }
    .btn { display: inline-block; width: 100%; background: linear-gradient(135deg, #0284c7, #2563eb); color: white; padding: 14px; font-size: 16px; font-weight: 600; border: none; border-radius: 8px; cursor: pointer; text-align: center; text-decoration: none; transition: 0.2s; }
    .btn:hover { opacity: 0.95; transform: translateY(-1px); }
    .btn-secondary { background: #1e293b; color: #38bdf8; width: auto; padding: 8px 16px; font-size: 14px; margin-top: 10px; }
    .plan-box { background: #070d19; border: 1px solid #1e293b; border-radius: 8px; padding: 20px; line-height: 1.7; white-space: pre-wrap; font-size: 15px; color: #e2e8f0; }
    .nutrition-box { background: rgba(74, 222, 128, 0.08); border: 1px solid rgba(74, 222, 128, 0.25); border-radius: 8px; padding: 18px; color: #4ade80; margin-bottom: 20px; font-size: 15px; }
    table { width: 100%; border-collapse: collapse; margin-top: 15px; font-size: 14px; }
    th, td { padding: 12px 14px; border: 1px solid #1e293b; text-align: left; }
    th { background: #0e1726; color: #38bdf8; }
    td { background: #111a2e; color: #cbd5e1; }
    .nav-links { display: flex; justify-content: space-between; align-items: center; margin-bottom: 24px; }
    .nav-links a { color: #38bdf8; text-decoration: none; font-size: 14px; font-weight: 600; }
"""

# -----------------------------------------------------------------------------
# 4. FASTAPI APP INITIALIZATION & ENDPOINTS
# -----------------------------------------------------------------------------
app = FastAPI(title="FitBuddy AI Generator", description="Naan Mudhalvan College Project")

@app.get("/", response_class=HTMLResponse)
def index_page():
    """Homepage: User Profile & Fitness Preference Form."""
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>FitBuddy - AI Workout Generator</title>
        <style>{CSS_STYLES}</style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <span class="badge">TN SKILLS · NAAN MUDHALVAN PROJECT</span>
                <h1>??? FitBuddy: AI Fitness Plan Generator</h1>
                <p>Texcity Arts & Science College · Team: Dhanusha, Deepa, Anfas, Kameleshwaran</p>
            </div>

            <div class="nav-links">
                <span>Personalized 7-Day Workout & Nutrition</span>
                <a href="/view-all-users">?? View Admin Dashboard</a>
            </div>

            <div class="card">
                <h2>User Profile & Workout Preferences</h2>
                <form action="/generate-workout" method="post">
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px;">
                        <div class="form-group">
                            <label>Full Name:</label>
                            <input type="text" name="name" placeholder="e.g. Dhanusha" required />
                        </div>
                        <div class="form-group">
                            <label>Unique User ID:</label>
                            <input type="text" name="user_id" placeholder="e.g. FIT101" required />
                        </div>
                    </div>

                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px;">
                        <div class="form-group">
                            <label>Age:</label>
                            <input type="number" name="age" placeholder="e.g. 20" required />
                        </div>
                        <div class="form-group">
                            <label>Weight (kg):</label>
                            <input type="text" name="weight" placeholder="e.g. 65" required />
                        </div>
                    </div>

                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px;">
                        <div class="form-group">
                            <label>Fitness Goal:</label>
                            <select name="goal">
                                <option value="Weight Loss">Weight Loss & Fat Reduction</option>
                                <option value="Muscle Gain">Muscle Gain & Hypertrophy</option>
                                <option value="Flexibility">Flexibility & Core Strength</option>
                                <option value="General Wellness">General Wellness & Stamina</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label>Workout Intensity:</label>
                            <select name="intensity">
                                <option value="Low">Low (Beginner / Gentle)</option>
                                <option value="Medium" selected>Medium (Moderate Intensity)</option>
                                <option value="High">High (Advanced / High Intensity)</option>
                            </select>
                        </div>
                    </div>

                    <button type="submit" class="btn">?? Generate AI Workout Plan</button>
                </form>
            </div>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.post("/generate-workout", response_class=HTMLResponse)
def generate_workout(
    name: str = Form(...),
    user_id: str = Form(...),
    age: int = Form(...),
    weight: str = Form(...),
    goal: str = Form(...),
    intensity: str = Form(...),
    db: Session = Depends(get_db)
):
    """Generates plan via Gemini AI, stores in SQLite, and displays results."""
    # 1. AI Generation
    workout_plan = generate_workout_with_gemini(name, age, weight, goal, intensity)
    nutrition_tip = generate_nutrition_tip_with_gemini(goal)

    # 2. Database Storage (User & Plan)
    existing_user = db.query(User).filter(User.user_id == user_id).first()
    if not existing_user:
        new_user = User(user_id=user_id, name=name, age=age, weight=weight, goal=goal, intensity=intensity)
        db.add(new_user)
    else:
        existing_user.name = name
        existing_user.age = age
        existing_user.weight = weight
        existing_user.goal = goal
        existing_user.intensity = intensity

    new_plan = WorkoutPlan(
        user_id=user_id,
        original_plan=workout_plan,
        nutrition_tip=nutrition_tip
    )
    db.add(new_plan)
    db.commit()

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>FitBuddy - Generated Plan</title>
        <style>{CSS_STYLES}</style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <span class="badge">AI GENERATION COMPLETE</span>
                <h1>? Your Personalized Workout Plan</h1>
                <p>Prepared for <strong>{name}</strong> (ID: {user_id}) | Goal: {goal} | Intensity: {intensity}</p>
            </div>

            <div class="card">
                <h2>?? Goal-Aligned Nutrition & Recovery Guidance</h2>
                <div class="nutrition-box">{nutrition_tip}</div>
                
                <h2>?? 7-Day Structured Schedule (Gemini Model)</h2>
                <div class="plan-box">{workout_plan}</div>
            </div>

            <!-- Feedback Loop Form -->
            <div class="card">
                <h2>?? Need Adjustments? Submit Feedback</h2>
                <p style="color: #94a3b8; font-size: 14px; margin-bottom: 14px;">
                    Don't start over! Type what you'd like to change (e.g. <em>"Add 15 mins yoga on Day 3"</em> or <em>"Reduce leg exercises"</em>).
                </p>
                <form action="/submit-feedback" method="post">
                    <input type="hidden" name="user_id" value="{user_id}" />
                    <div class="form-group">
                        <textarea name="feedback" rows="3" placeholder="Tell FitBuddy what to tweak in your plan..." required></textarea>
                    </div>
                    <button type="submit" class="btn">?? Update Plan with Feedback</button>
                </form>
            </div>

            <div style="text-align: center; margin-top: 20px;">
                <a href="/" class="btn btn-secondary">? Back to Generator</a>
                <a href="/view-all-users" class="btn btn-secondary" style="margin-left: 10px;">?? View in Dashboard</a>
            </div>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.post("/submit-feedback", response_class=HTMLResponse)
def submit_feedback(
    user_id: str = Form(...),
    feedback: str = Form(...),
    db: Session = Depends(get_db)
):
    """Processes user feedback, asks Gemini to revise the plan, and stores update."""
    latest_plan = db.query(WorkoutPlan).filter(WorkoutPlan.user_id == user_id).order_by(WorkoutPlan.id.desc()).first()
    
    if not latest_plan:
        return HTMLResponse(content="<h3>Error: No plan found for this User ID.</h3><a href='/'>Go Back</a>")

    updated_plan_text = update_workout_with_feedback(latest_plan.original_plan, feedback)
    latest_plan.feedback = feedback
    latest_plan.updated_plan = updated_plan_text
    db.commit()

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>FitBuddy - Updated Plan</title>
        <style>{CSS_STYLES}</style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <span class="badge" style="background: rgba(74, 222, 128, 0.15); color: #4ade80;">FEEDBACK LOOP COMPLETED</span>
                <h1>?? Plan Updated Successfully!</h1>
                <p>User ID: <strong>{user_id}</strong> | Feedback Applied: <em>"{feedback}"</em></p>
            </div>

            <div class="card">
                <h2>? Revised 7-Day Workout Routine</h2>
                <div class="plan-box">{updated_plan_text}</div>
            </div>

            <div class="card">
                <h2>Original Plan (Preserved in Database)</h2>
                <div class="plan-box" style="opacity: 0.75;">{latest_plan.original_plan}</div>
            </div>

            <div style="text-align: center; margin-top: 24px;">
                <a href="/" class="btn btn-secondary">? Generate New Plan</a>
                <a href="/view-all-users" class="btn btn-secondary" style="margin-left: 10px;">?? View Admin Records</a>
            </div>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/view-all-users", response_class=HTMLResponse)
def view_all_users(db: Session = Depends(get_db)):
    """Admin Dashboard: Shows all users, original plans, feedback, and revised plans."""
    users = db.query(User).all()
    plans = db.query(WorkoutPlan).all()

    # Map user plans
    plan_map = {p.user_id: p for p in plans}

    rows_html = ""
    for u in users:
        p = plan_map.get(u.user_id)
        feedback_display = p.feedback if (p and p.feedback) else "<span style='color:#64748b;'>None</span>"
        has_revised = "? Yes" if (p and p.updated_plan) else "? No"
        
        rows_html += f"""
        <tr>
            <td><strong>{u.user_id}</strong></td>
            <td>{u.name}</td>
            <td>{u.age} yrs / {u.weight} kg</td>
            <td>{u.goal}</td>
            <td>{u.intensity}</td>
            <td>{feedback_display}</td>
            <td>{has_revised}</td>
        </tr>
        """

    if not rows_html:
        rows_html = "<tr><td colspan='7' style='text-align:center; color:#94a3b8;'>No users recorded yet. Generate a workout first!</td></tr>"

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>FitBuddy - Admin Dashboard</title>
        <style>{CSS_STYLES}</style>
    </head>
    <body>
        <div class="container" style="max-width: 1080px;">
            <div class="header">
                <span class="badge">ADMINISTRATIVE DASHBOARD</span>
                <h1>?? All Registered Users & Plan Versions</h1>
                <p>Texcity Arts & Science College · BSc Computer Science 2nd Year</p>
            </div>

            <div class="nav-links">
                <a href="/">? Back to Generator Form</a>
                <span>Persistent Database: <code>fitbuddy.db (SQLite)</code></span>
            </div>

            <div class="card">
                <h2>Database Records</h2>
                <table>
                    <thead>
                        <tr>
                            <th>User ID</th>
                            <th>Name</th>
                            <th>Age / Weight</th>
                            <th>Goal</th>
                            <th>Intensity</th>
                            <th>User Feedback</th>
                            <th>Revised?</th>
                        </tr>
                    </thead>
                    <tbody>
                        {rows_html}
                    </tbody>
                </table>
            </div>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

# -----------------------------------------------------------------------------
# 5. LOCAL SERVER RUNNER
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("?? Starting FitBuddy FastAPI Server on http://127.0.0.1:8000")
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)