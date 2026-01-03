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

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="Face Wellness Tracker API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MongoDB connection
MONGO_URL = os.getenv("MONGO_URL")
DB_NAME = os.getenv("DB_NAME", "FaceWellness")
client = MongoClient(MONGO_URL)
db = client[DB_NAME]

# Collections
users_collection = db["users"]
sessions_collection = db["sessions"]
facial_analysis_collection = db["facial_analysis"]
habits_collection = db["habits"]

# External API configuration
AILAB_API_KEY = os.getenv("AILAB_API_KEY")
AILAB_API_URL = "https://www.ailabapi.com/api/portrait/analysis/skin-analysis-advanced"

# Google OAuth configuration
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")

# Session verification helper
async def verify_session(session_token: Optional[str] = Header(None)):
    if not session_token:
        raise HTTPException(status_code=401, detail="No session token provided")
    
    session = sessions_collection.find_one({"session_token": session_token})
    if not session:
        raise HTTPException(status_code=401, detail="Invalid session token")
    
    # Check if session is expired (7 days)
    if datetime.utcnow() > session["expires_at"]:
        sessions_collection.delete_one({"session_token": session_token})
        raise HTTPException(status_code=401, detail="Session expired")
    
    return session["user_id"]

# Health check endpoint
@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

# Google OAuth authentication endpoint
@app.post("/api/auth/google")
async def google_auth(request: dict):
    """
    Authenticate user with Google OAuth token
    Expected request body: {"credential": "google_jwt_token"}
    """
    try:
        # Get the credential from request
        credential = request.get("credential")
        if not credential:
            raise HTTPException(status_code=400, detail="No credential provided")
        
        # Verify the Google token
        try:
            idinfo = id_token.verify_oauth2_token(
                credential, 
                google_requests.Request(), 
                GOOGLE_CLIENT_ID
            )
        except ValueError as e:
            raise HTTPException(status_code=401, detail=f"Invalid token: {str(e)}")
        
        # Extract user information from the token
        email = idinfo.get("email")
        name = idinfo.get("name")
        picture = idinfo.get("picture")
        
        if not email:
            raise HTTPException(status_code=400, detail="Email not found in token")
        
        # Check if user exists
        existing_user = users_collection.find_one({"email": email})
        
        if existing_user:
            user_id = existing_user["user_id"]
        else:
            # Create new user
            user_id = str(uuid.uuid4())
            new_user = {
                "user_id": user_id,
                "email": email,
                "name": name,
                "picture": picture,
                "created_at": datetime.utcnow(),
                "total_photos": 0,
                "current_streak": 0,
                "longest_streak": 0
            }
            users_collection.insert_one(new_user)
        
        # Create session token
        session_token = str(uuid.uuid4())
        session_data = {
            "session_token": session_token,
            "user_id": user_id,
            "created_at": datetime.utcnow(),
            "expires_at": datetime.utcnow() + timedelta(days=7)
        }
        sessions_collection.insert_one(session_data)
        
        # Get user data
        user = users_collection.find_one({"user_id": user_id})
        user.pop("_id", None)
        
        return {
            "session_token": session_token,
            "user": user
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Authentication failed: {str(e)}")

# Get user profile
@app.get("/api/user/profile")
async def get_user_profile(user_id: str = Depends(verify_session)):
    """Get user profile with recent analysis data"""
    user = users_collection.find_one({"user_id": user_id})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    user.pop("_id", None)
    
    # Get recent analyses (last 7 days)
    seven_days_ago = datetime.utcnow() - timedelta(days=7)
    recent_analyses = list(facial_analysis_collection.find(
        {"user_id": user_id, "timestamp": {"$gte": seven_days_ago}}
    ).sort("timestamp", -1).limit(7))
    
    # Remove MongoDB _id and image data
    for analysis in recent_analyses:
        analysis.pop("_id", None)
        analysis.pop("image_base64", None)
    
    # Calculate current streak
    all_analyses = list(facial_analysis_collection.find(
        {"user_id": user_id}
    ).sort("timestamp", -1))
    
    current_streak = 0
    if all_analyses:
        last_date = all_analyses[0]["timestamp"].date()
        today = datetime.utcnow().date()
        
        if last_date == today:
            current_streak = 1
            for i in range(1, len(all_analyses)):
                expected_date = last_date - timedelta(days=i)
                if all_analyses[i]["timestamp"].date() == expected_date:
                    current_streak += 1
                else:
                    break
    
    user["current_streak"] = current_streak
    users_collection.update_one(
        {"user_id": user_id},
        {"$set": {"current_streak": current_streak}}
    )
    
    return {
        "user": user,
        "recent_analyses": recent_analyses
    }

# Analyze face endpoint - FIXED for AILab API
@app.post("/api/analyze-face")
async def analyze_face(
    image: UploadFile = File(...),
    user_id: str = Depends(verify_session)
):
    """Analyze uploaded face image using AILab API"""
    try:
        # Check if user already uploaded a photo today
        today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        today_end = today_start + timedelta(days=1)
        
        existing_analysis = facial_analysis_collection.find_one({
            "user_id": user_id,
            "timestamp": {"$gte": today_start, "$lt": today_end}
        })
        
        if existing_analysis:
            raise HTTPException(
                status_code=400,
                detail="You've already uploaded a photo today. Come back tomorrow!"
            )
        
        # Validate image type
        if image.content_type not in ["image/jpeg", "image/jpg", "image/png"]:
            raise HTTPException(status_code=400, detail="Invalid image type. Please upload JPG or PNG.")
        
        # Read image data
        image_data = await image.read()
        
        # Process image with PIL to ensure it's valid and convert to JPEG if needed
        try:
            img = Image.open(BytesIO(image_data))
            if img.mode != "RGB":
                img = img.convert("RGB")
            
            # Save as JPEG for consistency
            buffered = BytesIO()
            img.save(buffered, format="JPEG", quality=90)
            processed_image_data = buffered.getvalue()
            
            # Also save base64 for storage
            image_base64 = base64.b64encode(processed_image_data).decode()
            
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image: {str(e)}")
        
        # Call AILab API using multipart/form-data (CORRECT FORMAT)
        headers = {
            "ailabapi-api-key": AILAB_API_KEY
        }
        
        # Send as multipart/form-data with file
        files = {
            "image": ("face.jpg", processed_image_data, "image/jpeg")
        }
        
        print(f"Calling AILab API with key: {AILAB_API_KEY[:10]}...")
        
        response = requests.post(
            AILAB_API_URL, 
            headers=headers, 
            files=files
        )
        
        print(f"AILab API response status: {response.status_code}")
        print(f"AILab API response: {response.text[:500]}")
        
        if response.status_code != 200:
            raise HTTPException(
                status_code=500,
                detail=f"Face analysis failed: {response.text}"
            )
        
        ailab_result = response.json()
        
        # Check for API errors
        if ailab_result.get("error_code") != 0:
            error_msg = ailab_result.get("error_msg", "Unknown error")
            error_detail = ailab_result.get("error_detail", {})
            raise HTTPException(
                status_code=500,
                detail=f"Face analysis failed: {error_msg} - {error_detail}"
            )
        
        # Extract result data (AILab returns 'result' not 'data')
        result_data = ailab_result.get("result", {})
        
        # Store analysis in database
        analysis_id = str(uuid.uuid4())
        analysis_data = {
            "analysis_id": analysis_id,
            "user_id": user_id,
            "timestamp": datetime.utcnow(),
            "image_base64": image_base64,
            "results": {
                "eye_pouch": {
                    "value": result_data.get("eye_pouch", {}).get("value", 0),
                    "confidence": result_data.get("eye_pouch", {}).get("confidence", 0)
                },
                "dark_circle": {
                    "value": result_data.get("dark_circle", {}).get("value", 0),
                    "confidence": result_data.get("dark_circle", {}).get("confidence", 0)
                },
                "skin_age": {
                    "value": result_data.get("skin_age", {}).get("value", 0)
                },
                "forehead_wrinkle": {
                    "value": result_data.get("forehead_wrinkle", {}).get("value", 0),
                    "confidence": result_data.get("forehead_wrinkle", {}).get("confidence", 0)
                },
                "skin_color": result_data.get("skin_color", {}),
                "skin_type": result_data.get("skin_type", {}),
                "blackhead": result_data.get("blackhead", {}),
                "acne": result_data.get("acne", {}),
                "pores_forehead": result_data.get("pores_forehead", {}),
                "crows_feet": result_data.get("crows_feet", {}),
                "nasolabial_fold": result_data.get("nasolabial_fold", {})
            },
            "raw_result": result_data  # Store full result for future use
        }
        
        facial_analysis_collection.insert_one(analysis_data)
        
        # Update user stats
        users_collection.update_one(
            {"user_id": user_id},
            {"$inc": {"total_photos": 1}}
        )
        
        # Return analysis results
        return {
            "analysis_id": analysis_id,
            "timestamp": analysis_data["timestamp"].isoformat(),
            "results": analysis_data["results"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

# Log habits endpoint
@app.post("/api/habits/log")
async def log_habits(
    habits_data: dict,
    user_id: str = Depends(verify_session)
):
    """Log daily habits"""
    try:
        today = datetime.utcnow().date()
        
        # Update or create habit log for today
        habits_collection.update_one(
            {"user_id": user_id, "date": today.isoformat()},
            {"$set": {
                "user_id": user_id,
                "date": today.isoformat(),
                "habits": habits_data,
                "updated_at": datetime.utcnow()
            }},
            upsert=True
        )
        
        return {"message": "Habits logged successfully"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to log habits: {str(e)}")

# Get analysis history
@app.get("/api/analysis/history")
async def get_analysis_history(
    user_id: str = Depends(verify_session),
    limit: int = 30
):
    """Get user's analysis history"""
    try:
        analyses = list(facial_analysis_collection.find(
            {"user_id": user_id}
        ).sort("timestamp", -1).limit(limit))
        
        # Remove MongoDB _id and image data for performance
        for analysis in analyses:
            analysis.pop("_id", None)
            analysis.pop("image_base64", None)
            analysis.pop("raw_result", None)
        
        return {"history": analyses}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get history: {str(e)}")

# Get insights
@app.get("/api/insights")
async def get_insights(user_id: str = Depends(verify_session)):
    """Get personalized insights based on analysis history"""
    try:
        # Get last 7 analyses
        analyses = list(facial_analysis_collection.find(
            {"user_id": user_id}
        ).sort("timestamp", -1).limit(7))
        
        if not analyses:
            return {
                "message": "No analysis data available yet",
                "insights": []
            }
        
        # Calculate averages
        avg_eye_pouch = sum(a["results"]["eye_pouch"].get("value", 0) for a in analyses) / len(analyses)
        avg_dark_circle = sum(a["results"]["dark_circle"].get("value", 0) for a in analyses) / len(analyses)
        avg_skin_age = sum(float(a["results"]["skin_age"].get("value", 0)) for a in analyses) / len(analyses)
        
        insights = []
        
        # Generate insights based on values (0 = none, 1 = present for eye_pouch/forehead_wrinkle)
        # For dark_circle: 0 = none, 1-3 = different types
        if avg_eye_pouch > 0.5:
            insights.append({
                "type": "warning",
                "message": "Eye bags detected in recent photos. Consider getting more sleep and staying hydrated."
            })
        
        if avg_dark_circle > 0.5:
            insights.append({
                "type": "warning",
                "message": "Dark circles detected. Try using a vitamin C serum and ensure adequate rest."
            })
        
        if avg_skin_age > 35:
            insights.append({
                "type": "info",
                "message": f"Your estimated skin age is {int(avg_skin_age)}. Maintain a good skincare routine with sunscreen."
            })
        elif avg_skin_age > 0:
            insights.append({
                "type": "success",
                "message": f"Great! Your estimated skin age is {int(avg_skin_age)}. Keep up the good habits!"
            })
        
        if not insights:
            insights.append({
                "type": "success",
                "message": "Your skin metrics look healthy. Keep up the good work!"
            })
        
        return {
            "averages": {
                "eye_pouch": round(avg_eye_pouch, 2),
                "dark_circle": round(avg_dark_circle, 2),
                "skin_age": round(avg_skin_age, 1)
            },
            "insights": insights
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get insights: {str(e)}")

# Run the app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
