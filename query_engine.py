import google.generativeai as genai

# Load Gemini model
genai.configure(api_key="Gemini Key Here") 
model = genai.GenerativeModel("gemini-2.5-flash")

def ask_gemini(prompt):
    response = model.generate_content(prompt)
    return response.text.strip()
