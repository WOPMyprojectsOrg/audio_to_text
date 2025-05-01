import whisper
from pydub import AudioSegment
import time
import os

# Путь к аудиофайлу
audio_path = "audio.wav"
output_txt = "output.txt"
chunk_length_ms = 60000 * 30  # 10 секунд

# Загрузка модели
print("🧠 Загружаем модель Whisper...")
model = whisper.load_model("small")

# Загрузка аудио
print(f"🔊 Загружаем аудиофайл: {audio_path}")
audio = AudioSegment.from_file(audio_path)
total_length_ms = len(audio)
print(f"⏱️ Длина аудиофайла: {total_length_ms // 1000} секунд")

# Разбиваем на чанки
chunks = [audio[i:i + chunk_length_ms] for i in range(0, total_length_ms, chunk_length_ms)]
total_chunks = len(chunks)
transcribed_text = ""

# Сохранение временного файла для чанка
temp_dir = "temp_chunks"
os.makedirs(temp_dir, exist_ok=True)

# Обработка чанков
for i, chunk in enumerate(chunks):
    chunk_path = f"{temp_dir}/chunk_{i}.wav"
    chunk.export(chunk_path, format="wav")

    print(f"🔄 Обрабатываем чанк {i+1} из {total_chunks}...")

    result = model.transcribe(chunk_path, language="ru", fp16=False)
    transcribed_text += result["text"] + " "

    # Прогресс в %
    progress = (i + 1) / total_chunks * 100
    print(f"✅ Прогресс: {progress:.1f}%")

# Вывод результата
print("\n📝 Итоговый распознанный текст:")
print(transcribed_text.strip())

# Сохранение результата в txt-файл
with open(output_txt, "w", encoding="utf-8") as f:
    f.write(transcribed_text.strip())

print(f"\n📄 Текст успешно сохранён в файл: {output_txt}")