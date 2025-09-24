# AI Q&A Chat App with Streaming Responses

A Flutter frontend and Python backend application that provides real-time AI-powered Q&A with streaming responses.

## Project Structure
ai_qa_chat_app/
├── backend/ # Python FastAPI backend
└── frontend/ # Flutter application


## Setup Instructions

### Backend Setup

1. **Navigate to the backend directory:**
   ```bash
   cd backend
   
2. Create a virtual environment:
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   
4. Set up environment variables:
   Create a .env file in the backend directory:
   OPENAI_API_KEY=your_openai_api_key_here

5. Run the backend server:
   cd app
   python main.py


### Frontend Setup
1. Navigate to the frontend directory:
   cd frontend

2. Install Flutter dependencies:
    flutter pub get

3. Update the backend URL (if needed):
    In lib/providers/chat_provider.dart, update the base URL:

4. Run the Flutter app:
    flutter run

