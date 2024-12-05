#télécharger ffmpeg > l'encodage et le décodage audio, l'encapsulation et l'extraction de l'audio

# import pandas as pd
import whisper
from jiwer import wer

# le fichier TSV
tsv_file = "C:\\Users\\ananb\\Downloads\\cv-corpus-18.0-delta-2024-06-14\\fr\\validated.tsv"
df = pd.read_csv(tsv_file, sep='\t')

# modèle Whisper
model = whisper.load_model("tiny")

wer_scores = []

# décoder et évaluer 
for i in range(100):
    audio_path = f"C:\\Users\\ananb\\Downloads\\cv-corpus-18.0-delta-2024-06-14\\fr\\clips\\{df.iloc[i]['path']}"
    transcription_reference = df.iloc[i]['sentence']
    
    # décoder - whisper
    result = model.transcribe(audio_path)
    transcription_generated = result['text']
    
    # calcul - WER
    wer_score = wer(transcription_reference, transcription_generated)
    
    # calcul nmb de mots
    
    wer_scores.append(wer_score)

    # print
    print(f"Fichier {i+1}: WER = {wer_score:.2f}")
    print(f"Transcription de référence: {transcription_reference}")
    print(f"Transcription générée: {transcription_generated}")
    print("-" * 50)

# WER 


average_wer = sum(wer_scores) / len(wer_scores) #?
print(f"\nWER pour les 100 premiers fichiers: {average_wer:.2f}")

