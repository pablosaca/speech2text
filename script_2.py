import logging
import whisper
import pandas as pd


model = whisper.load_model('large')  # cargamos el modelo grande de Whisper
logging.info("Modelo whisper cargado")

# como sabemos que el audio está en castellano indicamos el lenguaje
response = model.transcribe(
    audio="audios/youtube/quillo_barrios_campeones_liga.mp3",
    language="es",
    verbose=True
)

d = {"texto": response["text"]}
df = pd.DataFrame(d, index=[0])

df.to_csv("textos/real_madrid_gana_liga.csv")
