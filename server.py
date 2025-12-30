from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pymongo import MongoClient
from datetime import datetime, timedelta
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests
import requests
import os
import base64
import io
from PIL import Image
import uuid
import json
import secrets
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Face Wellness Tracker API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MongoDB connection
client = MongoClient(os.getenv("MONGO_URL"))
db = client[os.getenv("DB_NAME")]
users_collection = db.users
sessions_collection = db.sessions
analysis_collection = db.facial_analysis
habits_collection = db.habits

# API Configuration
AILAB_API_KEY = os.getenv("AILAB_API_KEY")
AILAB_API_URL = "https://www.ailabapi.com/api/portrait/analysis/skin-analysis-advanced"
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")

# Helper function to verify session
async def verify_session(session_token: str = Header(None)):
    if not session_token:
        raise HTTPException(status_code=401, detail="Session token required")
    
    session = sessions_collection.find_one({"session_token": session_token})
    if not session:
        raise HTTPException(status_code=401, detail="Invalid session")
    
    # Check if session is expired
    if datetime.utcnow() > session["expires_at"]:
        sessions_collection.delete_one({"session_token": session_token})
        raise HTTPException(status_code=401, detail="Session expired")
    
    return session["user_id"]

@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow()}

@app.post("/api/auth/google")
async def google_auth(request: dict):
    """Handle Google OAuth authentication"""
    try:
        token = request.get("credential")
        if not token:
            raise HTTPException(status_code=400, detail="Google credential required")
        
        # Verify Google token
        try:
            idinfo = id_token.verify_oauth2_token(
                token, 
                google_requests.Request(), 
                GOOGLE_CLIENT_ID
            )
            
            if idinfo['iss'] not in ['accounts.google.com', 'https://accounts.google.com']:
                raise ValueError('Wrong issuer.')
                
        except ValueError as e:
            raise HTTPException(status_code=401, detail=f"Invalid token: {str(e)}")
        
        # Extract user info
        email = idinfo['email']
        name = idinfo.get('name', '')
        picture = idinfo.get('picture', '')
        
        # Check if user exists
        existing_user = users_collection.find_one({"email": email})
        
        if not existing_user:
            # Create new user
            user_id = str(uuid.uuid4())
            user_doc = {
                "user_id": user_id,
                "email": email,
                "name": name,
                "picture": picture,
                "created_at": datetime.utcnow(),
                "total_photos": 0,
                "current_streak": 0,
                "longest_streak": 0,
                "last_photo_date": None
            }
            users_collection.insert_one(user_doc)
        else:
            user_id = existing_user["user_id"]
            name = existing_user["name"]
            picture = existing_user["picture"]
        
        # Create session
        session_token = secrets.token_urlsafe(32)
        session_doc = {
            "session_token": session_token,
            "user_id": user_id,
            "created_at": datetime.utcnow(),
            "expires_at": datetime.utcnow() + timedelta(days=30)
        }
        sessions_collection.insert_one(session_doc)
        
        # Get user stats
        user = users_collection.find_one({"user_id": user_id})
        
        return {
            "session_token": session_token,
            "user": {
                "user_id": user_id,
                "email": email,
                "name": name,
                "picture": picture,
                "total_photos": user["total_photos"],
                "current_streak": user["current_streak"],
                "longest_streak": user["longest_streak"]
            }
        }
        
    except Exception as e:
        print(f"Auth error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/user/profile")
async def get_user_profile(user_id: str = Depends(verify_session)):
    """Get user profile and stats"""
    user = users_collection.find_one({"user_id": user_id})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Get recent analysis
    recent_analysis = list(analysis_collection.find(
        {"user_id": user_id}
    ).sort("timestamp", -1).limit(7))
    
    # Calculate streak
    today = datetime.utcnow().date()
    streak = 0
    for analysis in recent_analysis:
        analysis_date = analysis["timestamp"].date()
        days_diff = (today - analysis_date).days
        if days_diff == streak:
            streak += 1
        else:
            break
    
    return {
        "user": {
            "user_id": user["user_id"],
            "email": user["email"],
            "name": user["name"],
            "picture": user.get("picture", ""),
            "total_photos": user.get("total_photos", 0),
            "current_streak": streak,
            "longest_streak": user.get("longest_streak", 0)
        },
        "recent_analysis": recent_analysis
    }

@app.post("/api/analyze-face")
async def analyze_face(
    image: UploadFile = File(...),
    user_id: str = Depends(verify_session)
):
    """Analyze uploaded face image"""
    try:
        # Check if image was uploaded today
        today = datetime.utcnow().date()
        existing_today = analysis_collection.find_one({
            "user_id": user_id,
            "timestamp": {
                "$gte": datetime.combine(today, datetime.min.time()),
                "$lt": datetime.combine(today + timedelta(days=1), datetime.min.time())
            }
        })
        
        if existing_today:
            raise HTTPException(status_code=400, detail="Photo already uploaded today")
        
        # Validate image
        if not image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read and process image
        contents = await image.read()
        
        # Convert to base64 for storage
        image_base64 = base64.b64encode(contents).decode('utf-8')
        
        # Prepare request to AILabTools API
        files = {"image": (image.filename, contents, image.content_type)}
        headers = {"ailabapi-api-key": AILAB_API_KEY}
        
        # Call AILabTools API
        response = requests.post(AILAB_API_URL, files=files, headers=headers)
        
        if response.status_code != 200:
            raise HTTPException(status_code=500, detail="Facial analysis failed")
        
        api_result = response.json()
        
        if api_result.get("error_code") != 0:
            raise HTTPException(status_code=400, detail=api_result.get("error_msg", "API error"))
        
        # Extract analysis results
        result = api_result.get("result", {})
        
        analysis_data = {
            "analysis_id": str(uuid.uuid4()),
            "user_id": user_id,
            "timestamp": datetime.utcnow(),
            "image_base64": image_base64,
            "results": {
                "eye_pouch": {
                    "value": result.get("eye_pouch", {}).get("value", 0),
                    "confidence": result.get("eye_pouch", {}).get("confidence", 0)
                },
                "dark_circle": {
                    "value": result.get("dark_circle", {}).get("value", 0),
                    "confidence": result.get("dark_circle", {}).get("confidence", 0)
                },
                "skin_age": {
                    "value": result.get("skin_age", {}).get("value", 25)
                },
                "forehead_wrinkle": {
                    "value": result.get("forehead_wrinkle", {}).get("value", 0),
                    "confidence": result.get("forehead_wrinkle", {}).get("confidence", 0)
                }
            }
        }
        
        # Save analysis
        analysis_collection.insert_one(analysis_data)
        
        # Update user stats
        users_collection.update_one(
            {"user_id": user_id},
            {"$inc": {"total_photos": 1}}
        )
        
        return {
            "analysis_id": analysis_data["analysis_id"],
            "timestamp": analysis_data["timestamp"],
            "results": analysis_data["results"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/habits/log")
async def log_daily_habits(
    habits_data: dict,
    user_id: str = Depends(verify_session)
):
    """Log daily habits"""
    try:
        today = datetime.utcnow().date()
        
        # Check if habits already logged today
        existing_habits = habits_collection.find_one({
            "user_id": user_id,
            "date": today
        })
        
        if existing_habits:
            # Update existing habits
            habits_collection.update_one(
                {"user_id": user_id, "date": today},
                {"$set": {
                    "habits": habits_data,
                    "updated_at": datetime.utcnow()
                }}
            )
        else:
            # Create new habits entry
            habits_doc = {
                "user_id": user_id,
                "date": today,
                "habits": habits_data,
                "created_at": datetime.utcnow()
            }
            habits_collection.insert_one(habits_doc)
        
        return {"message": "Habits logged successfully"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/analysis/history")
async def get_analysis_history(
    limit: int = 30,
    user_id: str = Depends(verify_session)
):
    """Get user's analysis history"""
    try:
        history = list(analysis_collection.find(
            {"user_id": user_id},
            {"image_base64": 0}  # Exclude image data for performance
        ).sort("timestamp", -1).limit(limit))
        
        return {"history": history}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/insights")
async def get_insights(user_id: str = Depends(verify_session)):
    """Generate insights from user's data"""
    try:
        # Get recent analysis (last 7 days)
        recent_analysis = list(analysis_collection.find(
            {"user_id": user_id}
        ).sort("timestamp", -1).limit(7))
        
        if not recent_analysis:
            return {"message": "No data available for insights"}
        
        # Calculate averages
        avg_eye_pouch = sum(a["results"]["eye_pouch"]["value"] for a in recent_analysis) / len(recent_analysis)
        avg_dark_circle = sum(a["results"]["dark_circle"]["value"] for a in recent_analysis) / len(recent_analysis)
        avg_skin_age = sum(a["results"]["skin_age"]["value"] for a in recent_analysis) / len(recent_analysis)
        
        # Generate insights
        insights = []
        
        if avg_eye_pouch > 1.5:
            insights.append({
                "type": "warning",
                "message": "Your eye puffiness levels are elevated. Consider getting more sleep and reducing screen time."
            })
        
        if avg_dark_circle > 1.5:
            insights.append({
                "type": "warning", 
                "message": "Dark circles are prominent. Stay hydrated and maintain a consistent sleep schedule."
            })
        
        if avg_skin_age > 30:
            insights.append({
                "type": "info",
                "message": "Consider a skincare routine with moisturizer and sunscreen for healthier skin."
            })
        
        if not insights:
            insights.append({
                "type": "success",
                "message": "Great job! Your facial wellness indicators are looking good. Keep up the healthy habits!"
            })
        
        return {
            "insights": insights,
            "averages": {
                "eye_pouch": round(avg_eye_pouch, 2),
                "dark_circle": round(avg_dark_circle, 2),
                "skin_age": round(avg_skin_age, 1)
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
