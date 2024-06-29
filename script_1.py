import os
import logging
import json
import whisper

path = "audios/sentiment_analysis"
output_path = "textos/speech2text_audios.json"

model = whisper.load_model('large')
logging.info("Modelo whisper cargado")

audio_dict = {}
for key in os.listdir(path):
    if os.path.isfile(os.path.join(path, key)):
        try:
            audio_dict[f"audio_{key}"] = model.transcribe(
                audio=os.path.join(path, key),
                language="en",
                verbose=True)["text"]
        except Exception as e:
            logging.info(e)

# Guardar el diccionario como JSON en un archivo
with open(output_path, "w") as file:
    json.dump(audio_dict, file)
