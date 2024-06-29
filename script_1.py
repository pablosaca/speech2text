import os
import logging
import json
import whisper

# Configuración básica del logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

path = "audios/sentiment_analysis"
output_path = "textos/speech2text_audios_ejemplo.json"

model = whisper.load_model('large')
logging.info("Modelo whisper cargado")

audio_dict = {}
for key in os.listdir(path):
    if os.path.isfile(os.path.join(path, key)):
        if key == "1.wav":
            try:
                audio_dict[f"audio_{key}"] = model.transcribe(
                    audio=os.path.join(path, key),
                    language="en",
                    verbose=True)["text"]
                logger.info(f"Audio {key} transcrito")
            except Exception as e:
                logger.info(e)

# Guardar el diccionario como JSON en un archivo
with open(output_path, "w") as file:
    json.dump(audio_dict, file)
