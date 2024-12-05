#decode whisper > modèle de reconnaissance vocale d'OpenAI
import whisper

model = whisper.load_model("tiny")
result = model.transcribe("c:\\Users\\ananb\\Downloads\\cv-corpus-18.0-delta-2024-06-14\\fr\\clips\\common_voice_fr_40859724.mp3")

print(result["text"])

#PS C:\Windows\system32> $addPath = "C:\Users\ananb\Downloads\ffmpeg-6.1-essentials_build\bin\ffmpeg.exe"