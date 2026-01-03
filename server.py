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

# AILab has multiple skin analysis endpoints - try them in order of preference
AILAB_ENDPOINTS = {
    "skin_analysis": "https://www.ailabapi.com/api/portrait/analysis/skin-analysis",
    "skin_analysis_advanced": "https://www.ailabapi.com/api/portrait/analysis/skin-analysis-advanced",
    "face_analyzer": "https://www.ailabapi.com/api/portrait/analysis/face-analyzer"
}

# Google OAuth configuration
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")

# Session verification helper
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

# Health check endpoint
@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy", 
        "timestamp": datetime.utcnow().isoformat(),
        "api_key_set": bool(AILAB_API_KEY),
        "db_connected": bool(MONGO_URL)
    }

# Google OAuth authentication endpoint
@app.post("/api/auth/google")
async def google_auth(request: dict):
    try:
        credential = request.get("credential")
        if not credential:
            raise HTTPException(status_code=400, detail="No credential provided")
        
        try:
            idinfo = id_token.verify_oauth2_token(
                credential, 
                google_requests.Request(), 
                GOOGLE_CLIENT_ID
            )
        except ValueError as e:
            raise HTTPException(status_code=401, detail=f"Invalid token: {str(e)}")
        
        email = idinfo.get("email")
        name = idinfo.get("name")
        picture = idinfo.get("picture")
        
        if not email:
            raise HTTPException(status_code=400, detail="Email not found in token")
        
        existing_user = users_collection.find_one({"email": email})
        
        if existing_user:
            user_id = existing_user["user_id"]
        else:
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
        
        session_token = str(uuid.uuid4())
        session_data = {
            "session_token": session_token,
            "user_id": user_id,
            "created_at": datetime.utcnow(),
            "expires_at": datetime.utcnow() + timedelta(days=7)
        }
        sessions_collection.insert_one(session_data)
        
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
    user = users_collection.find_one({"user_id": user_id})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    user.pop("_id", None)
    
    seven_days_ago = datetime.utcnow() - timedelta(days=7)
    recent_analyses = list(facial_analysis_collection.find(
        {"user_id": user_id, "timestamp": {"$gte": seven_days_ago}}
    ).sort("timestamp", -1).limit(7))
    
    for analysis in recent_analyses:
        analysis.pop("_id", None)
        analysis.pop("image_base64", None)
    
    all_analyses = list(facial_analysis_collection.find(
        {"user_id": user_id}
    ).sort("timestamp", -1))
    
    current_streak = 0
    if all_analyses:
        last_date = all_analyses[0]["timestamp"].date()
        today = datetime.utcnow().date()
        
        if last_date == today or last_date == today - timedelta(days=1):
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


def call_ailab_api(image_data: bytes) -> dict:
    """
    Try to call AILab API with multiple endpoints as fallback
    """
    headers = {
        "ailabapi-api-key": AILAB_API_KEY
    }
    
    # Prepare the file for upload
    files = {
        "image": ("face.jpg", image_data, "image/jpeg")
    }
    
    errors = []
    
    # Try skin-analysis endpoint first (basic, most reliable)
    for endpoint_name, endpoint_url in AILAB_ENDPOINTS.items():
        try:
            print(f"Trying {endpoint_name}: {endpoint_url}")
            
            # Reset file pointer for each attempt
            files = {
                "image": ("face.jpg", image_data, "image/jpeg")
            }
            
            response = requests.post(
                endpoint_url,
                headers=headers,
                files=files,
                timeout=30
            )
            
            print(f"Response status: {response.status_code}")
            print(f"Response body: {response.text[:500]}")
            
            if response.status_code == 200:
                result = response.json()
                
                # Check for API-level errors
                if result.get("error_code") == 0:
                    print(f"Success with {endpoint_name}")
                    return {
                        "success": True,
                        "endpoint": endpoint_name,
                        "data": result
                    }
                else:
                    error_msg = result.get("error_msg", "Unknown error")
                    errors.append(f"{endpoint_name}: {error_msg}")
            else:
                errors.append(f"{endpoint_name}: HTTP {response.status_code} - {response.text[:200]}")
                
        except requests.exceptions.Timeout:
            errors.append(f"{endpoint_name}: Timeout")
        except Exception as e:
            errors.append(f"{endpoint_name}: {str(e)}")
    
    # All endpoints failed
    return {
        "success": False,
        "errors": errors
    }


def extract_skin_results(api_response: dict, endpoint: str) -> dict:
    """
    Extract and normalize skin analysis results from different API endpoints
    """
    # Default values
    results = {
        "eye_pouch": {"value": 0, "confidence": 0},
        "dark_circle": {"value": 0, "confidence": 0},
        "skin_age": {"value": 0},
        "forehead_wrinkle": {"value": 0, "confidence": 0},
        "skin_color": {"value": 0, "confidence": 0},
        "skin_type": {"skin_type": 0},
        "acne": {},
        "blackhead": {"value": 0, "confidence": 0}
    }
    
    try:
        # skin-analysis and skin-analysis-advanced return "result" directly
        if endpoint in ["skin_analysis", "skin_analysis_advanced"]:
            data = api_response.get("result", {})
        # face-analyzer returns data differently
        elif endpoint == "face_analyzer":
            face_infos = api_response.get("face_detail_infos", [])
            if face_infos:
                data = face_infos[0].get("face_detail_attributes_info", {})
                # Map face analyzer fields to our format
                return {
                    "eye_pouch": {"value": 0, "confidence": 0},
                    "dark_circle": {"value": 0, "confidence": 0},
                    "skin_age": {"value": data.get("age", 0)},
                    "forehead_wrinkle": {"value": 0, "confidence": 0},
                    "skin_color": {"value": 0, "confidence": 0},
                    "skin_type": {"skin_type": 0},
                    "acne": {},
                    "blackhead": {"value": 0, "confidence": 0}
                }
            return results
        else:
            data = api_response.get("result", api_response.get("data", {}))
        
        # Map the results
        if "eye_pouch" in data:
            results["eye_pouch"] = {
                "value": data["eye_pouch"].get("value", 0),
                "confidence": data["eye_pouch"].get("confidence", 0)
            }
        
        if "dark_circle" in data:
            results["dark_circle"] = {
                "value": data["dark_circle"].get("value", 0),
                "confidence": data["dark_circle"].get("confidence", 0)
            }
        
        if "skin_age" in data:
            results["skin_age"] = {
                "value": data["skin_age"].get("value", 0)
            }
        
        if "forehead_wrinkle" in data:
            results["forehead_wrinkle"] = {
                "value": data["forehead_wrinkle"].get("value", 0),
                "confidence": data["forehead_wrinkle"].get("confidence", 0)
            }
        
        if "skin_color" in data:
            results["skin_color"] = data["skin_color"]
        
        if "skin_type" in data:
            results["skin_type"] = data["skin_type"]
        
        if "blackhead" in data:
            results["blackhead"] = data["blackhead"]
        
        if "acne" in data:
            results["acne"] = data["acne"]
            
    except Exception as e:
        print(f"Error extracting results: {e}")
    
    return results


# Analyze face endpoint
@app.post("/api/analyze-face")
async def analyze_face(
    image: UploadFile = File(...),
    user_id: str = Depends(verify_session)
):
    """Analyze uploaded face image using AILab API"""
    try:
        # Check daily limit
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
        
        # Read and process image
        image_data = await image.read()
        
        try:
            img = Image.open(BytesIO(image_data))
            
            # Validate image size
            if img.size[0] < 200 or img.size[1] < 200:
                raise HTTPException(status_code=400, detail="Image too small. Minimum 200x200 pixels required.")
            
            if img.size[0] > 4096 or img.size[1] > 4096:
                # Resize if too large
                img.thumbnail((4096, 4096), Image.Resampling.LANCZOS)
            
            # Convert to RGB if needed
            if img.mode != "RGB":
                img = img.convert("RGB")
            
            # Save as high-quality JPEG
            buffered = BytesIO()
            img.save(buffered, format="JPEG", quality=95)
            processed_image_data = buffered.getvalue()
            
            # Base64 for storage
            image_base64 = base64.b64encode(processed_image_data).decode()
            
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image: {str(e)}")
        
        # Check API key
        if not AILAB_API_KEY:
            raise HTTPException(status_code=500, detail="API key not configured. Please contact support.")
        
        # Call AILab API
        api_result = call_ailab_api(processed_image_data)
        
        if not api_result["success"]:
            error_details = "; ".join(api_result.get("errors", ["Unknown error"]))
            raise HTTPException(
                status_code=500,
                detail=f"Face analysis failed: {error_details}"
            )
        
        # Extract results
        endpoint_used = api_result["endpoint"]
        api_data = api_result["data"]
        results = extract_skin_results(api_data, endpoint_used)
        
        # Store analysis
        analysis_id = str(uuid.uuid4())
        analysis_data = {
            "analysis_id": analysis_id,
            "user_id": user_id,
            "timestamp": datetime.utcnow(),
            "image_base64": image_base64,
            "results": results,
            "api_endpoint": endpoint_used,
            "raw_response": api_data  # Store for debugging
        }
        
        facial_analysis_collection.insert_one(analysis_data)
        
        # Update user stats
        users_collection.update_one(
            {"user_id": user_id},
            {"$inc": {"total_photos": 1}}
        )
        
        return {
            "analysis_id": analysis_id,
            "timestamp": analysis_data["timestamp"].isoformat(),
            "results": results
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Analysis error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


# Log habits endpoint
@app.post("/api/habits/log")
async def log_habits(
    habits_data: dict,
    user_id: str = Depends(verify_session)
):
    try:
        today = datetime.utcnow().date()
        
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
    try:
        analyses = list(facial_analysis_collection.find(
            {"user_id": user_id}
        ).sort("timestamp", -1).limit(limit))
        
        for analysis in analyses:
            analysis.pop("_id", None)
            analysis.pop("image_base64", None)
            analysis.pop("raw_response", None)
        
        return {"history": analyses}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get history: {str(e)}")


# Get insights
@app.get("/api/insights")
async def get_insights(user_id: str = Depends(verify_session)):
    try:
        analyses = list(facial_analysis_collection.find(
            {"user_id": user_id}
        ).sort("timestamp", -1).limit(7))
        
        if not analyses:
            return {
                "message": "No analysis data available yet",
                "insights": []
            }
        
        # Calculate averages
        eye_pouch_values = [a["results"].get("eye_pouch", {}).get("value", 0) for a in analyses]
        dark_circle_values = [a["results"].get("dark_circle", {}).get("value", 0) for a in analyses]
        skin_age_values = [a["results"].get("skin_age", {}).get("value", 0) for a in analyses]
        
        avg_eye_pouch = sum(eye_pouch_values) / len(eye_pouch_values) if eye_pouch_values else 0
        avg_dark_circle = sum(dark_circle_values) / len(dark_circle_values) if dark_circle_values else 0
        avg_skin_age = sum(skin_age_values) / len(skin_age_values) if skin_age_values else 0
        
        insights = []
        
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
        
        if avg_skin_age > 0:
            if avg_skin_age > 40:
                insights.append({
                    "type": "info",
                    "message": f"Your estimated skin age is {int(avg_skin_age)}. Consider adding retinol and sunscreen to your routine."
                })
            else:
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


# Debug endpoint to test API connection
@app.get("/api/debug/test-ailab")
async def test_ailab():
    """Test endpoint to verify AILab API connection"""
    if not AILAB_API_KEY:
        return {"error": "AILAB_API_KEY not set"}
    
    return {
        "api_key_prefix": AILAB_API_KEY[:10] + "...",
        "api_key_length": len(AILAB_API_KEY),
        "endpoints_configured": list(AILAB_ENDPOINTS.keys())
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
