"""
Face Wellness Tracker - Backend API v5
Features:
- AILab Skin Analysis Advanced API integration
- Comprehensive metric extraction (20+ metrics)
- Instant analysis + stored results
- Weekly email reports ready
- Streak and badge system
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

app = FastAPI(title="Face Wellness Tracker API v5")

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

# AILab API
AILAB_API_KEY = os.getenv("AILAB_API_KEY")
AILAB_API_URL = "https://www.ailabapi.com/api/portrait/analysis/skin-analysis-advanced"

# Google OAuth
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")


async def verify_session(session_token: Optional[str] = Header(None)):
    if not session_token:
        raise HTTPException(status_code=401, detail="No session token provided")
    
    session = sessions_collection.find_one({"session_token": session_token})
    if not session:
        raise HTTPException(status_code=401, detail="Invalid session token")
    
    if datetime.utcnow() > session["expires_at"]:
        sessions_collection.delete_one({"session_token": session_token})
        raise HTTPException(status_code=401, detail="Session expired")
    
    return session["user_id"]


def calculate_streak(user_id: str) -> tuple:
    """Calculate current and longest streak"""
    analyses = list(facial_analysis_collection.find(
        {"user_id": user_id}
    ).sort("timestamp", -1))
    
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
    longest_streak = max(user.get("longest_streak", 0), current_streak)
    
    return current_streak, longest_streak


def analyze_with_ailab(image_data: bytes) -> dict:
    """
    Call AILab Skin Analysis Advanced API
    Returns ALL available metrics
    """
    if not AILAB_API_KEY:
        return {"success": False, "error": "AILAB_API_KEY not configured"}
    
    headers = {
        "ailabapi-api-key": AILAB_API_KEY
    }
    
    files = {
        "image": ("face.jpg", image_data, "image/jpeg")
    }
    
    # Request additional data
    data = {
        "return_rect_confidence": "1"  # Get confidence for acne/spots
    }
    
    try:
        print(f"Calling AILab API...")
        response = requests.post(
            AILAB_API_URL,
            headers=headers,
            files=files,
            data=data,
            timeout=60
        )
        
        print(f"AILab Response Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"AILab Response: {result.get('error_code')}, {result.get('error_msg')}")
            
            if result.get("error_code") == 0:
                api_result = result.get("result", {})
                
                # Extract ALL metrics from AILab response
                extracted = {
                    # Skin basics
                    "skin_age": api_result.get("skin_age", {}),
                    "skin_color": api_result.get("skin_color", {}),
                    "skin_type": api_result.get("skin_type", {}),
                    "skintone_ita": api_result.get("skintone_ita", {}),
                    "skin_hue_ha": api_result.get("skin_hue_ha", {}),
                    
                    # Eye area
                    "eye_pouch": api_result.get("eye_pouch", {}),
                    "eye_pouch_severity": api_result.get("eye_pouch_severity", {}),
                    "dark_circle": api_result.get("dark_circle", {}),
                    "crows_feet": api_result.get("crows_feet", {}),
                    "eye_finelines": api_result.get("eye_finelines", {}),
                    "left_eyelids": api_result.get("left_eyelids", {}),
                    "right_eyelids": api_result.get("right_eyelids", {}),
                    
                    # Wrinkles
                    "forehead_wrinkle": api_result.get("forehead_wrinkle", {}),
                    "glabella_wrinkle": api_result.get("glabella_wrinkle", {}),
                    "nasolabial_fold": api_result.get("nasolabial_fold", {}),
                    "nasolabial_fold_severity": api_result.get("nasolabial_fold_severity", {}),
                    
                    # Pores
                    "pores_forehead": api_result.get("pores_forehead", {}),
                    "pores_left_cheek": api_result.get("pores_left_cheek", {}),
                    "pores_right_cheek": api_result.get("pores_right_cheek", {}),
                    "pores_jaw": api_result.get("pores_jaw", {}),
                    
                    # Skin issues
                    "blackhead": api_result.get("blackhead", {}),
                    "acne": api_result.get("acne", {}),
                    "mole": api_result.get("mole", {}),
                    "closed_comedones": api_result.get("closed_comedones", {}),
                    "skin_spot": api_result.get("skin_spot", {}),
                    
                    # Sensitivity (if available)
                    "sensitivity": api_result.get("sensitivity", {}),
                    
                    # Face rectangle (for reference)
                    "face_rectangle": result.get("face_rectangle", {}),
                    
                    # Warnings
                    "warnings": result.get("warning", [])
                }
                
                return {
                    "success": True,
                    "results": extracted
                }
            else:
                return {
                    "success": False,
                    "error": f"AILab error: {result.get('error_msg', 'Unknown error')}"
                }
        else:
            return {
                "success": False,
                "error": f"HTTP {response.status_code}: {response.text[:200]}"
            }
            
    except requests.exceptions.Timeout:
        return {"success": False, "error": "Request timeout - try again"}
    except Exception as e:
        print(f"AILab exception: {e}")
        return {"success": False, "error": str(e)}


# ============================================
# ROUTES
# ============================================

@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "ailab_configured": bool(AILAB_API_KEY),
        "db_connected": bool(MONGO_URL)
    }


@app.post("/api/auth/google")
async def google_auth(request: dict):
    try:
        credential = request.get("credential")
        if not credential:
            raise HTTPException(status_code=400, detail="No credential")
        
        idinfo = id_token.verify_oauth2_token(
            credential, google_requests.Request(), GOOGLE_CLIENT_ID
        )
        
        email = idinfo.get("email")
        name = idinfo.get("name")
        picture = idinfo.get("picture")
        
        if not email:
            raise HTTPException(status_code=400, detail="Email not found")
        
        existing = users_collection.find_one({"email": email})
        
        if existing:
            user_id = existing["user_id"]
        else:
            user_id = str(uuid.uuid4())
            users_collection.insert_one({
                "user_id": user_id,
                "email": email,
                "name": name,
                "picture": picture,
                "created_at": datetime.utcnow(),
                "total_photos": 0,
                "current_streak": 0,
                "longest_streak": 0
            })
        
        session_token = str(uuid.uuid4())
        sessions_collection.insert_one({
            "session_token": session_token,
            "user_id": user_id,
            "created_at": datetime.utcnow(),
            "expires_at": datetime.utcnow() + timedelta(days=7)
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
    
    recent = list(facial_analysis_collection.find(
        {"user_id": user_id}
    ).sort("timestamp", -1).limit(7))
    
    for r in recent:
        r.pop("_id", None)
        r.pop("image_base64", None)
    
    return {"user": user, "recent_analyses": recent}


@app.post("/api/analyze-face")
async def analyze_face(
    image: UploadFile = File(...),
    user_id: str = Depends(verify_session)
):
    """Analyze face with AILab - returns comprehensive metrics"""
    try:
        # Check daily limit
        today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        today_end = today_start + timedelta(days=1)
        
        existing = facial_analysis_collection.find_one({
            "user_id": user_id,
            "timestamp": {"$gte": today_start, "$lt": today_end}
        })
        
        if existing:
            raise HTTPException(status_code=400, detail="Already analyzed today! Come back tomorrow.")
        
        # Validate
        if image.content_type not in ["image/jpeg", "image/jpg", "image/png"]:
            raise HTTPException(status_code=400, detail="Invalid image type")
        
        # Process image
        image_data = await image.read()
        
        try:
            img = Image.open(BytesIO(image_data))
            
            if img.size[0] < 200 or img.size[1] < 200:
                raise HTTPException(status_code=400, detail="Image too small (min 200x200)")
            
            if img.mode != "RGB":
                img = img.convert("RGB")
            
            # Resize if too large
            if img.size[0] > 4096 or img.size[1] > 4096:
                img.thumbnail((4096, 4096), Image.Resampling.LANCZOS)
            
            buffered = BytesIO()
            img.save(buffered, format="JPEG", quality=95)
            processed_data = buffered.getvalue()
            image_base64 = base64.b64encode(processed_data).decode()
            
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image: {e}")
        
        # Call AILab
        result = analyze_with_ailab(processed_data)
        
        if not result["success"]:
            raise HTTPException(status_code=500, detail=f"Analysis failed: {result.get('error')}")
        
        # Save to database
        analysis_id = str(uuid.uuid4())
        analysis_data = {
            "analysis_id": analysis_id,
            "user_id": user_id,
            "timestamp": datetime.utcnow(),
            "image_base64": image_base64,
            "results": result["results"]
        }
        
        facial_analysis_collection.insert_one(analysis_data)
        
        # Update user stats
        users_collection.update_one(
            {"user_id": user_id},
            {"$inc": {"total_photos": 1}}
        )
        
        # Update streak
        current_streak, longest_streak = calculate_streak(user_id)
        users_collection.update_one(
            {"user_id": user_id},
            {"$set": {"current_streak": current_streak, "longest_streak": longest_streak}}
        )
        
        return {
            "analysis_id": analysis_id,
            "timestamp": analysis_data["timestamp"].isoformat(),
            "results": result["results"],
            "current_streak": current_streak,
            "longest_streak": longest_streak
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/analysis/history")
async def get_history(user_id: str = Depends(verify_session), limit: int = 30):
    analyses = list(facial_analysis_collection.find(
        {"user_id": user_id}
    ).sort("timestamp", -1).limit(limit))
    
    for a in analyses:
        a.pop("_id", None)
        a.pop("image_base64", None)
    
    return {"history": analyses}


@app.get("/api/insights")
async def get_insights(user_id: str = Depends(verify_session)):
    analyses = list(facial_analysis_collection.find(
        {"user_id": user_id}
    ).sort("timestamp", -1).limit(7))
    
    if not analyses:
        return {"message": "No data yet", "insights": [], "averages": {}}
    
    # Calculate averages
    skin_ages = [a["results"].get("skin_age", {}).get("value", 0) for a in analyses if a.get("results")]
    eye_pouches = [a["results"].get("eye_pouch", {}).get("value", 0) for a in analyses if a.get("results")]
    dark_circles = [a["results"].get("dark_circle", {}).get("value", 0) for a in analyses if a.get("results")]
    blackheads = [a["results"].get("blackhead", {}).get("value", 0) for a in analyses if a.get("results")]
    
    avg_age = sum(skin_ages) / len(skin_ages) if skin_ages else 0
    avg_eye = sum(eye_pouches) / len(eye_pouches) if eye_pouches else 0
    avg_dark = sum(dark_circles) / len(dark_circles) if dark_circles else 0
    avg_blackhead = sum(blackheads) / len(blackheads) if blackheads else 0
    
    insights = []
    
    if avg_eye > 0.5:
        insights.append({"type": "warning", "message": "Eye bags detected in recent photos. Consider getting more sleep and staying hydrated."})
    
    if avg_dark > 0:
        dark_types = ["", "Pigmented dark circles detected - try vitamin C serum.", 
                      "Vascular dark circles detected - try cold compresses and more sleep.",
                      "Shadow-type dark circles detected - consider filler or lifestyle changes."]
        most_common = max(set([a["results"].get("dark_circle", {}).get("value", 0) for a in analyses if a.get("results")]), key=[a["results"].get("dark_circle", {}).get("value", 0) for a in analyses if a.get("results")].count)
        if most_common > 0 and most_common < len(dark_types):
            insights.append({"type": "warning", "message": dark_types[most_common]})
    
    if avg_blackhead > 1:
        insights.append({"type": "warning", "message": "Blackheads detected. Try salicylic acid cleanser and regular exfoliation."})
    
    # Check for improvements
    if len(analyses) >= 2:
        latest = analyses[0]["results"]
        previous = analyses[1]["results"]
        
        if latest.get("skin_age", {}).get("value", 100) < previous.get("skin_age", {}).get("value", 0):
            insights.append({"type": "success", "message": "Your skin age improved since last scan! Keep up the good work."})
    
    if not insights:
        insights.append({"type": "success", "message": "Your skin looks healthy! Keep up your routine."})
    
    return {
        "averages": {
            "skin_age": round(avg_age, 1),
            "eye_pouch": round(avg_eye, 2),
            "dark_circle": round(avg_dark, 2),
            "blackhead": round(avg_blackhead, 2)
        },
        "insights": insights,
        "total_scans": len(analyses)
    }


@app.get("/api/debug/test-ailab")
async def test_ailab():
    """Debug endpoint to verify AILab configuration"""
    return {
        "api_key_configured": bool(AILAB_API_KEY),
        "api_key_prefix": AILAB_API_KEY[:10] + "..." if AILAB_API_KEY else None,
        "api_key_length": len(AILAB_API_KEY) if AILAB_API_KEY else 0,
        "endpoint": AILAB_API_URL
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
