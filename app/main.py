from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List
import groq
import json
import os
from app.config import settings

app = FastAPI(title="AI Q&A Chat API", version="1.0.0")

# CORS middleware to allow Flutter app to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic model for request validation
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]

# Initialize Groq client
def get_groq_client():
    if not settings.GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="Groq API key not configured")
    return groq.Groq(api_key=settings.GROQ_API_KEY)

@app.get("/")
async def root():
    return {"message": "AI Q&A Chat API is running"}

@app.post("/chat")
async def chat_stream(chat_request: ChatRequest):
    """
    Stream AI responses using Groq API
    Expects messages in format: [{"role": "user", "content": "message"}]
    """
    try:
        client = get_groq_client()
        
        # Convert to format expected by Groq API
        messages = [{"role": msg.role, "content": msg.content} for msg in chat_request.messages]
        
        print(f"Received {len(messages)} messages from client")
        
        # Create chat completion with streaming
        chat_completion = client.chat.completions.create(
            messages=messages,
            model=settings.MODEL_NAME,
            stream=True,
            temperature=0.7,
            max_tokens=1024
        )
        
        async def generate():
            for chunk in chat_completion:
                if chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    # Format as proper SSE
                    data = json.dumps({"content": content})
                    yield f"data: {data}\n\n"
            
            # Send completion signal
            yield "data: [DONE]\n\n"
        
        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "*",
            }
        )
    
    except groq.APIError as e:
        print(f"Groq API error: {e}")
        raise HTTPException(status_code=500, detail=f"Groq API error: {str(e)}")
    except Exception as e:
        print(f"Internal server error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "AI Q&A Chat API"}