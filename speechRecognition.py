import speech_recognition as sr
recognizer = sr.Recognizer()
mic = sr.Microphone()

def recognizeSpeech():
    while True:
        try:
            with sr.Microphone() as source2:
                r.adjust_for_ambient_noise(source2, duration=0.2)
                audio2 = r.listen(source2)

            recognizedText = r.recognize_google(audio2)
            recognizedText = recognizedText.lower()
            return recognizedText

        except sr.RequestError as e:
            print("Speech recognition failed:", e)
        
        except sr.UnknownValueError:
            print("Could not understand audio")
