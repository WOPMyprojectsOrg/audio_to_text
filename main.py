import speech_recognition as sr

recognizer = sr.Recognizer()
audio_file = "short.wav"

with sr.AudioFile(audio_file) as source:
    audio_data = recognizer.record(source)
    text = recognizer.recognize_sphinx(audio_data, language='ru')
    print(text)