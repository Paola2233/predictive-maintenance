import vertexai
from vertexai.generative_models import GenerativeModel

# Reemplaza con tus datos
PROJECT_ID = "abiding-kingdom-491823-b0"
LOCATION = "us-central1" # Una de las regiones más baratas

vertexai.init(project=PROJECT_ID, location=LOCATION)

model = GenerativeModel("gemini-2.5-flash")

response = model.generate_content("Dame una idea de proyecto simple para aprender Vertex AI")

print(f"Respuesta de Vertex AI: {response.text}")