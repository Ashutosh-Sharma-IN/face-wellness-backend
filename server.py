"""
Face Wellness Tracker - Backend API v6
Features:
- AILab Skin Analysis (20+ metrics)
- Habit Tracking & Logging
- Correlation Analysis (habits vs face metrics)
- AI-powered Insights
- Weekly Reports Ready
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from pymongo import MongoClient
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
from PIL import Image
import requests
import uuid
import base64
from io import BytesIO
from typing import Optional
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests

load_dotenv()

app = FastAPI(title="Face Wellness Tracker API v6")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MongoDB
MONGO_URL = os.getenv("MONGO_URL")
DB_NAME = os.getenv("DB_NAME", "FaceWellness")
client = MongoClient(MONGO_URL)
db = client[DB_NAME]

users_collection = db["users"]
sessions_collection = db["sessions"]
facial_analysis_collection = db["facial_analysis"]
habits_collection = db["habits"]

# AILab
AILAB_API_KEY = os.getenv("AILAB_API_KEY")
AILAB_API_URL = "https://www.ailabapi.com/api/portrait/analysis/skin-analysis-advanced"

# Google OAuth
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")


async def verify_session(session_token: Optional[str] = Header(None)):
    if not session_token:
        raise HTTPException(status_code=401, detail="No session token")
    session = sessions_collection.find_one({"session_token": session_token})
    if not session:
        raise HTTPException(status_code=401, detail="Invalid session")
    if datetime.utcnow() > session["expires_at"]:
        sessions_collection.delete_one({"session_token": session_token})
        raise HTTPException(status_code=401, detail="Session expired")
    return session["user_id"]


def calculate_streak(user_id: str) -> tuple:
    analyses = list(facial_analysis_collection.find({"user_id": user_id}).sort("timestamp", -1))
    if not analyses:
        return 0, 0
    
    current_streak = 0
    today = datetime.utcnow().date()
    last_date = analyses[0]["timestamp"].date()
    
    if last_date == today or last_date == today - timedelta(days=1):
        current_streak = 1
        check_date = last_date
        for analysis in analyses[1:]:
            expected = check_date - timedelta(days=1)
            analysis_date = analysis["timestamp"].date()
            if analysis_date == expected:
                current_streak += 1
                check_date = analysis_date
            elif analysis_date == check_date:
                continue
            else:
                break
    
    user = users_collection.find_one({"user_id": user_id})
    longest = max(user.get("longest_streak", 0), current_streak)
    return current_streak, longest


def analyze_with_ailab(image_data: bytes) -> dict:
    if not AILAB_API_KEY:
        return {"success": False, "error": "API key not configured"}
    
    headers = {"ailabapi-api-key": AILAB_API_KEY}
    files = {"image": ("face.jpg", image_data, "image/jpeg")}
    
    try:
        response = requests.post(AILAB_API_URL, headers=headers, files=files, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            if result.get("error_code") == 0:
                api_result = result.get("result", {})
                return {
                    "success": True,
                    "results": {
                        "skin_age": api_result.get("skin_age", {}),
                        "skin_color": api_result.get("skin_color", {}),
                        "skin_type": api_result.get("skin_type", {}),
                        "eye_pouch": api_result.get("eye_pouch", {}),
                        "eye_pouch_severity": api_result.get("eye_pouch_severity", {}),
                        "dark_circle": api_result.get("dark_circle", {}),
                        "forehead_wrinkle": api_result.get("forehead_wrinkle", {}),
                        "crows_feet": api_result.get("crows_feet", {}),
                        "eye_finelines": api_result.get("eye_finelines", {}),
                        "glabella_wrinkle": api_result.get("glabella_wrinkle", {}),
                        "nasolabial_fold": api_result.get("nasolabial_fold", {}),
                        "pores_forehead": api_result.get("pores_forehead", {}),
                        "pores_left_cheek": api_result.get("pores_left_cheek", {}),
                        "pores_right_cheek": api_result.get("pores_right_cheek", {}),
                        "pores_jaw": api_result.get("pores_jaw", {}),
                        "blackhead": api_result.get("blackhead", {}),
                        "acne": api_result.get("acne", {}),
                        "mole": api_result.get("mole", {}),
                        "skin_spot": api_result.get("skin_spot", {})
                    }
                }
            return {"success": False, "error": result.get("error_msg", "Unknown")}
        return {"success": False, "error": f"HTTP {response.status_code}"}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ============================================
# ROUTES
# ============================================

@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}


@app.post("/api/auth/google")
async def google_auth(request: dict):
    try:
        credential = request.get("credential")
        if not credential:
            raise HTTPException(status_code=400, detail="No credential")
        
        idinfo = id_token.verify_oauth2_token(credential, google_requests.Request(), GOOGLE_CLIENT_ID)
        email, name, picture = idinfo.get("email"), idinfo.get("name"), idinfo.get("picture")
        
        if not email:
            raise HTTPException(status_code=400, detail="Email not found")
        
        existing = users_collection.find_one({"email": email})
        if existing:
            user_id = existing["user_id"]
        else:
            user_id = str(uuid.uuid4())
            users_collection.insert_one({
                "user_id": user_id, "email": email, "name": name, "picture": picture,
                "created_at": datetime.utcnow(), "total_photos": 0, "current_streak": 0, "longest_streak": 0
            })
        
        session_token = str(uuid.uuid4())
        sessions_collection.insert_one({
            "session_token": session_token, "user_id": user_id,
            "created_at": datetime.utcnow(), "expires_at": datetime.utcnow() + timedelta(days=7)
        })
        
        user = users_collection.find_one({"user_id": user_id})
        user.pop("_id", None)
        return {"session_token": session_token, "user": user}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/user/profile")
async def get_user_profile(user_id: str = Depends(verify_session)):
    user = users_collection.find_one({"user_id": user_id})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    user.pop("_id", None)
    current_streak, longest_streak = calculate_streak(user_id)
    user["current_streak"] = current_streak
    user["longest_streak"] = longest_streak
    
    users_collection.update_one(
        {"user_id": user_id},
        {"$set": {"current_streak": current_streak, "longest_streak": longest_streak}}
    )
    
    recent = list(facial_analysis_collection.find({"user_id": user_id}).sort("timestamp", -1).limit(7))
    for r in recent:
        r.pop("_id", None)
        r.pop("image_base64", None)
    
    return {"user": user, "recent_analyses": recent}


@app.post("/api/analyze-face")
async def analyze_face(image: UploadFile = File(...), user_id: str = Depends(verify_session)):
    try:
        # Check daily limit
        today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        existing = facial_analysis_collection.find_one({
            "user_id": user_id,
            "timestamp": {"$gte": today_start, "$lt": today_start + timedelta(days=1)}
        })
        if existing:
            raise HTTPException(status_code=400, detail="Already analyzed today! Come back tomorrow.")
        
        # Process image
        image_data = await image.read()
        img = Image.open(BytesIO(image_data))
        if img.mode != "RGB":
            img = img.convert("RGB")
        if img.size[0] > 4096 or img.size[1] > 4096:
            img.thumbnail((4096, 4096), Image.Resampling.LANCZOS)
        
        buffered = BytesIO()
        img.save(buffered, format="JPEG", quality=95)
        processed = buffered.getvalue()
        
        # Analyze
        result = analyze_with_ailab(processed)
        if not result["success"]:
            raise HTTPException(status_code=500, detail=f"Analysis failed: {result.get('error')}")
        
        # Get today's habits if logged
        today_habits = habits_collection.find_one({
            "user_id": user_id,
            "date": datetime.utcnow().strftime("%Y-%m-%d")
        })
        
        # Save
        analysis_id = str(uuid.uuid4())
        facial_analysis_collection.insert_one({
            "analysis_id": analysis_id,
            "user_id": user_id,
            "timestamp": datetime.utcnow(),
            "image_base64": base64.b64encode(processed).decode(),
            "results": result["results"],
            "habits": today_habits.get("habits") if today_habits else None
        })
        
        users_collection.update_one({"user_id": user_id}, {"$inc": {"total_photos": 1}})
        
        current_streak, longest_streak = calculate_streak(user_id)
        users_collection.update_one(
            {"user_id": user_id},
            {"$set": {"current_streak": current_streak, "longest_streak": longest_streak}}
        )
        
        return {
            "analysis_id": analysis_id,
            "timestamp": datetime.utcnow().isoformat(),
            "results": result["results"],
            "current_streak": current_streak,
            "longest_streak": longest_streak
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/habits/log")
async def log_habits(habits_data: dict, user_id: str = Depends(verify_session)):
    """Log daily habits"""
    try:
        today = datetime.utcnow().strftime("%Y-%m-%d")
        
        habits_collection.update_one(
            {"user_id": user_id, "date": today},
            {"$set": {
                "user_id": user_id,
                "date": today,
                "habits": habits_data,
                "updated_at": datetime.utcnow()
            }},
            upsert=True
        )
        
        # Also update today's face analysis with habits if exists
        today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        facial_analysis_collection.update_one(
            {"user_id": user_id, "timestamp": {"$gte": today_start}},
            {"$set": {"habits": habits_data}}
        )
        
        return {"message": "Habits logged successfully", "date": today}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/habits/today")
async def get_today_habits(user_id: str = Depends(verify_session)):
    """Get today's logged habits"""
    today = datetime.utcnow().strftime("%Y-%m-%d")
    habits = habits_collection.find_one({"user_id": user_id, "date": today})
    
    if habits:
        habits.pop("_id", None)
        return habits
    return {"message": "No habits logged today"}


@app.get("/api/analysis/history")
async def get_history(user_id: str = Depends(verify_session), limit: int = 30):
    analyses = list(facial_analysis_collection.find({"user_id": user_id}).sort("timestamp", -1).limit(limit))
    for a in analyses:
        a.pop("_id", None)
        a.pop("image_base64", None)
    return {"history": analyses}


@app.get("/api/correlations")
async def get_correlations(user_id: str = Depends(verify_session)):
    """Analyze correlations between habits and face metrics"""
    
    # Get last 30 days of data
    analyses = list(facial_analysis_collection.find(
        {"user_id": user_id, "habits": {"$exists": True, "$ne": None}}
    ).sort("timestamp", -1).limit(30))
    
    if len(analyses) < 3:
        return {
            "message": "Need at least 3 days of combined face + habit data for correlations",
            "correlations": [],
            "data_points": len(analyses)
        }
    
    correlations = []
    
    # Calculate correlations
    sleep_data = [(a["habits"].get("sleep_hours", 0), a["results"].get("eye_pouch", {}).get("value", 0)) for a in analyses if a.get("habits")]
    water_data = [(a["habits"].get("water_glasses", 0), a["results"].get("skin_type", {}).get("skin_type", 2)) for a in analyses if a.get("habits")]
    stress_data = [(a["habits"].get("stress_level", 5), a["results"].get("forehead_wrinkle", {}).get("value", 0)) for a in analyses if a.get("habits")]
    
    # Sleep vs Eye Bags
    if sleep_data:
        low_sleep = [s for s in sleep_data if s[0] < 7]
        high_sleep = [s for s in sleep_data if s[0] >= 7]
        
        low_sleep_bags = sum(1 for s in low_sleep if s[1] == 1) / len(low_sleep) * 100 if low_sleep else 0
        high_sleep_bags = sum(1 for s in high_sleep if s[1] == 1) / len(high_sleep) * 100 if high_sleep else 0
        
        if low_sleep_bags > high_sleep_bags + 20:
            correlations.append({
                "habit": "sleep",
                "face_metric": "eye_bags",
                "strength": "strong",
                "finding": f"When you sleep <7 hours, you have eye bags {low_sleep_bags:.0f}% of the time vs {high_sleep_bags:.0f}% with 7+ hours",
                "recommendation": "Aim for 7-8 hours of sleep to reduce eye bags"
            })
    
    # Water vs Dry Skin
    if water_data:
        low_water = [w for w in water_data if w[0] < 6]
        high_water = [w for w in water_data if w[0] >= 6]
        
        low_water_dry = sum(1 for w in low_water if w[1] == 1) / len(low_water) * 100 if low_water else 0
        high_water_dry = sum(1 for w in high_water if w[1] == 1) / len(high_water) * 100 if high_water else 0
        
        if low_water_dry > high_water_dry + 15:
            correlations.append({
                "habit": "hydration",
                "face_metric": "dry_skin",
                "strength": "moderate",
                "finding": f"Low water intake correlates with dry skin ({low_water_dry:.0f}% vs {high_water_dry:.0f}%)",
                "recommendation": "Drink 8+ glasses of water daily"
            })
    
    # Stress vs Wrinkles
    if stress_data:
        high_stress = [s for s in stress_data if s[0] > 6]
        low_stress = [s for s in stress_data if s[0] <= 6]
        
        high_stress_wrinkles = sum(1 for s in high_stress if s[1] == 1) / len(high_stress) * 100 if high_stress else 0
        low_stress_wrinkles = sum(1 for s in low_stress if s[1] == 1) / len(low_stress) * 100 if low_stress else 0
        
        if high_stress_wrinkles > low_stress_wrinkles + 15:
            correlations.append({
                "habit": "stress",
                "face_metric": "forehead_wrinkles",
                "strength": "moderate",
                "finding": f"High stress days show more forehead lines ({high_stress_wrinkles:.0f}% vs {low_stress_wrinkles:.0f}%)",
                "recommendation": "Try stress-reduction techniques like meditation"
            })
    
    return {
        "correlations": correlations,
        "data_points": len(analyses),
        "message": f"Analysis based on {len(analyses)} days of data"
    }


@app.get("/api/insights")
async def get_insights(user_id: str = Depends(verify_session)):
    """Generate AI-powered insights"""
    
    analyses = list(facial_analysis_collection.find({"user_id": user_id}).sort("timestamp", -1).limit(7))
    
    if not analyses:
        return {"insights": [], "message": "No data yet"}
    
    insights = []
    
    # Calculate averages
    skin_ages = [a["results"].get("skin_age", {}).get("value", 0) for a in analyses]
    avg_age = sum(skin_ages) / len(skin_ages) if skin_ages else 0
    
    # Check trends
    if len(analyses) >= 2:
        latest = analyses[0]["results"]
        previous = analyses[1]["results"]
        
        # Improvement detection
        if latest.get("eye_pouch", {}).get("value", 1) < previous.get("eye_pouch", {}).get("value", 1):
            insights.append({
                "type": "positive",
                "icon": "✨",
                "message": "Your eye bags have improved since last scan!"
            })
        
        if latest.get("skin_age", {}).get("value", 100) < previous.get("skin_age", {}).get("value", 0):
            insights.append({
                "type": "positive", 
                "icon": "🎉",
                "message": f"Your skin age improved! Now showing {latest.get('skin_age', {}).get('value', 0)} years."
            })
    
    # Habit-based insights
    latest_habits = analyses[0].get("habits") if analyses else None
    if latest_habits:
        if latest_habits.get("sleep_hours", 8) < 7:
            insights.append({
                "type": "warning",
                "icon": "😴",
                "message": f"Only {latest_habits.get('sleep_hours')} hours of sleep. This may cause eye bags and dull skin."
            })
        
        if latest_habits.get("water_glasses", 8) < 6:
            insights.append({
                "type": "warning",
                "icon": "💧",
                "message": f"Low water intake ({latest_habits.get('water_glasses')} glasses). Hydration is key for skin health!"
            })
        
        if latest_habits.get("exercise_minutes", 0) >= 30:
            insights.append({
                "type": "positive",
                "icon": "💪",
                "message": f"Great job on {latest_habits.get('exercise_minutes')} minutes of exercise! This boosts skin circulation."
            })
    
    if not insights:
        insights.append({
            "type": "info",
            "icon": "💡",
            "message": "Keep tracking to discover patterns between your habits and skin health!"
        })
    
    return {
        "insights": insights,
        "averages": {
            "skin_age": round(avg_age, 1)
        },
        "data_points": len(analyses)
    }


@app.get("/api/weekly-report")
async def get_weekly_report(user_id: str = Depends(verify_session)):
    """Generate weekly report data"""
    
    week_ago = datetime.utcnow() - timedelta(days=7)
    
    analyses = list(facial_analysis_collection.find({
        "user_id": user_id,
        "timestamp": {"$gte": week_ago}
    }).sort("timestamp", 1))
    
    habits = list(habits_collection.find({
        "user_id": user_id,
        "date": {"$gte": week_ago.strftime("%Y-%m-%d")}
    }))
    
    if not analyses:
        return {"message": "No data for this week"}
    
    # Calculate weekly stats
    skin_ages = [a["results"].get("skin_age", {}).get("value", 0) for a in analyses]
    eye_bags_days = sum(1 for a in analyses if a["results"].get("eye_pouch", {}).get("value", 0) == 1)
    dark_circles_days = sum(1 for a in analyses if a["results"].get("dark_circle", {}).get("value", 0) > 0)
    
    avg_sleep = sum(h["habits"].get("sleep_hours", 0) for h in habits if h.get("habits")) / len(habits) if habits else 0
    avg_water = sum(h["habits"].get("water_glasses", 0) for h in habits if h.get("habits")) / len(habits) if habits else 0
    avg_exercise = sum(h["habits"].get("exercise_minutes", 0) for h in habits if h.get("habits")) / len(habits) if habits else 0
    
    return {
        "period": {
            "start": week_ago.isoformat(),
            "end": datetime.utcnow().isoformat()
        },
        "scans_completed": len(analyses),
        "face_metrics": {
            "avg_skin_age": round(sum(skin_ages) / len(skin_ages), 1) if skin_ages else 0,
            "skin_age_trend": skin_ages,
            "eye_bags_days": eye_bags_days,
            "dark_circles_days": dark_circles_days
        },
        "habit_averages": {
            "sleep_hours": round(avg_sleep, 1),
            "water_glasses": round(avg_water, 1),
            "exercise_minutes": round(avg_exercise, 0)
        },
        "habits_logged": len(habits)
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
