# Face Wellness Tracker - Backend API

FastAPI backend for facial wellness analysis.

## Environment Variables Required

Set these in your deployment platform (Railway/Render):

```
MONGO_URL=mongodb+srv://username:password@cluster.mongodb.net/
DB_NAME=face_wellness_tracker
AILAB_API_KEY=your_ailab_api_key_here
```

## Local Development

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Create `.env` file with the variables above

3. Run server:
   ```bash
   uvicorn server:app --reload
   ```

## API Endpoints

- `GET /api/health` - Health check
- `POST /api/auth/profile` - User authentication
- `GET /api/user/profile` - Get user profile
- `POST /api/analyze-face` - Analyze face photo
- `GET /api/insights` - Get personalized insights
- `GET /api/analysis/history` - Get analysis history

## Deployment

Deploy to Railway.app or Render.com
