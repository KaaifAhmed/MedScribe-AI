from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import google.generativeai as genai
import os
import shutil
import json

# --- CONFIGURATION ---
# ⚠️ PASTE YOUR API KEY HERE FOR LOCAL TESTING
if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = "AIzaSyAAlZyBbA4-eGoE6zm_GdLqCeL6IiQ7e1o"

genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the Frontend
@app.get("/")
async def read_index():
    return FileResponse('index.html')

@app.post("/transcribe")
def transcribe_audio(file: UploadFile = File(...)):
    print(f"📥 Received file: {file.file}")
    
    # 1. Save the uploaded file temporarily
    temp_filename = f"temp_{file.filename}"
    with open(temp_filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        # 2. Upload to Gemini
        print("🚀 Uploading to Gemini...")
        audio_file = genai.upload_file(path=temp_filename)
        
        # Wait for processing
        import time
        while audio_file.state.name == "PROCESSING":
            time.sleep(1)
            audio_file = genai.get_file(audio_file.name)

        # 3. Generate Analysis
        model = genai.GenerativeModel('models/gemini-flash-latest')
        
        prompt = """
        You are MedScribe-CS. Listen to this Code-Switched (Urdu/English) medical consultation.
        
        Task 1: Transcribe verbatim in Roman Urdu, with Urdu medical terms. Also diarize according to who is speaking, like doctor, patient, etc.
        Task 2: Extract a SOAP Note JSON, in the SOAP note, don't include any urdu terms.
        
        Output strictly valid JSON.
        
        Output Format:
        {
            "transcript": "...",
            "soap": { "subjective": "...", "objective": "...", "assessment": "...", "plan": "..." }
        }
        """
        
        response = model.generate_content(
            [prompt, audio_file],
            generation_config={"response_mime_type": "application/json"}
        )        
        
        # Cleanup temp file
        os.remove(temp_filename)
        
        # ... (inside transcribe_audio function) ...
        
        result_json = json.loads(response.text)
        
        # ADD THIS DEBUG PRINT:
        print("\n✅ Sending back JSON to Frontend:")
        print(json.dumps(result_json, indent=2)[:500] + "...") # Print first 500 chars
        
        # Cleanup temp file
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
        
        return result_json

    except Exception as e:
        return {"error": str(e)}

# Run with: uvicorn main:app --reload