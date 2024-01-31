from transformers import pipeline
import sounddevice as sd
import soundfile as sf
import numpy as np
import pyttsx3
import os
import time
import speech_recognition as sr

def record_audio(duration=6):
    # 初始化語音識別器
    recognizer = sr.Recognizer()

    # 設置麥克風作為音源
    microphone = sr.Microphone()

    with microphone as source:
        print("Say something within 6 seconds:")
        audio = recognizer.listen(source, timeout=duration)  # timeout=6 限定6秒內講完

    try:
        # 進行語音識別
        text_result = recognizer.recognize_google(audio, language="zh-TW")
        print("You said:", text_result)
        return text_result

    except sr.UnknownValueError:
        print("Could not understand audio.")
        return ""
    except sr.RequestError as e:
        print(f"Error with the request to Google Speech Recognition service: {e}")
        return ""


input("Press Enter and say 'hey sir' to activate: ")

while True:
    time.sleep(1)
    transcript = record_audio().lower()
    print("ASR Transcript:", transcript)
    print("Detected phrase:", transcript)

    if "hey sir" in transcript:
        print("Activation phrase detected. Start conversation.")
        conversation = []

        while True:
            # 提問
            user_input = input("You: ")

            # 將對話加入 conversation
            conversation.append("You: " + user_input)

            # ChatGPT回答
            chatgpt_response = chatgpt_pipe(conversation, max_length=100, num_return_sequences=1, no_repeat_ngram_size=2, top_k=50, top_p=0.95)
            response_text = chatgpt_response[0].get('generated_text', '')

            if response_text.strip() == "":
                print("AI: I'm sorry, I didn't understand your question.")
            else:
                print("AI:", response_text)

                # 將 AI 的回應轉成語音
                engine = pyttsx3.init()
                engine.save_to_file(response_text, 'ai_response.wav')
                engine.runAndWait()

                # 播放 AI 的回應
                os.system("start ai_response.wav")

            # 如果需要結束對話，可以加入條件判斷
            if "exit" in user_input.lower():
                break

            # 提問下一個問題
            print("Press Enter and ask your next question:")
            input()

    # 如果需要結束整個對話，可以加入條件判斷
    if "exit" in transcript.lower():
        break


