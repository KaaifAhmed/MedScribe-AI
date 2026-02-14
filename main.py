from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from google import genai
from dotenv import load_dotenv
import os
import shutil
import json

# --- CONFIGURATION ---

# 1. First, try to get the key from the System (Production/Render)
google_api_key = os.getenv("GOOGLE_API_KEY")

# 2. If not found, load the .env file and try again (Local Development)
if not google_api_key:
    print("🔍 Key not in system environment. Loading .env file...")
    load_dotenv() 
    google_api_key = os.getenv("GOOGLE_API_KEY")

# 3. Final Check: If it's STILL missing, stop the app.
if not google_api_key:
    raise ValueError(
        "❌ FATAL ERROR: GOOGLE_API_KEY is missing!\n"
        "   👉 On Localhost: Create a '.env' file with 'GOOGLE_API_KEY=your_key'.\n"
        "   👉 On Render: Go to Settings > Environment Variables and add it there."
    )

print("✅ API Key loaded successfully.")

client = genai.Client(api_key=google_api_key)

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
        audio_file = client.files.upload(file=temp_filename)
        
        # Wait for processing
        import time
        while audio_file.state.name == "PROCESSING":
            time.sleep(1)

            audio_file = client.files.get(audio_file.name)
            print("Processing...")
                
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

        print("Generating response...")
        response = client.models.generate_content(
            contents=[prompt, audio_file],
            model="models/gemini-flash-latest",
            config={'response_mime_type': 'application/json'},
        )

        print("Response generated!")
        print(response)
        
        # Cleanup temp file
        os.remove(temp_filename)
        
        # ... (inside transcribe_audio function) ...
        
        result_json = json.loads(response.text)
        
        # ADD THIS DEBUG PRINT:
        print(json.dumps(result_json, indent=2)[:500] + "...") # Print first 500 chars
        print("\n✅ Sending back JSON to Frontend:")    
        
        # Cleanup temp file
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
        
        return result_json  

    except Exception as e:
            # CRITICAL: Print the error so you can see it in the terminal
            print(f"❌ ERROR: {str(e)}")
            return {"error": str(e)}
        
    finally:
        # Cleanup temp file always
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
# Run with: uvicorn main:app --reload