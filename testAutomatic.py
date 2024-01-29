from transformers import pipeline

# 指定音訊檔案路徑
my_file_path = "C:\\Users\\fclin\\Desktop\\test0125\\01252.wav"

# 初始化 ASR 管道
pipe = pipeline("automatic-speech-recognition", model="openai/whisper-large-v2")

# 進行預測
results = pipe(my_file_path)

print(results)








