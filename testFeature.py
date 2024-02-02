import torch
import torchaudio
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import librosa.display
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
from scipy.fftpack import dct

class ConformerWithWav2Vec2(nn.Module):
    def __init__(self, conformer_model, wav2vec2_model):
        super(ConformerWithWav2Vec2, self).__init__()
        self.wav2vec2_model = wav2vec2_model
        self.conformer_model = conformer_model

    def forward(self, input_ids):
        # Use Wav2Vec2 model for feature extraction
        features = self.wav2vec2_model(input_ids).logits

        # Conformer model takes features as input
        output = self.conformer_model(features)
        return output

# 定義 ConformerModel
class ConformerModel(nn.Module):
    def __init__(self, input_size, d_model, n_head, feedforward_dim, num_layers, output_size, dropout_rate):
        super(ConformerModel, self).__init__()

        # Conformer 模型結構
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_head,
                dim_feedforward=feedforward_dim,
                dropout=dropout_rate
            ),
            num_layers=num_layers
        )
        self.fc = nn.Linear(d_model, output_size)

    def forward(self, x):
        # Conformer 模型的前向傳播
        x = self.transformer_encoder(x)
        # 全連接層的運算
        output = self.fc(x)
        return output

# 為語音情感分類定義模型的超參數
input_size = 80  # 設定特徵的維度
d_model = 32  # Conformer 模型的維度
n_head = 8  # TransformerEncoderLayer 的 head 數量
feedforward_dim = 2048  # TransformerEncoderLayer 的前向網絡的維度
num_layers = 6  # TransformerEncoder 的層數
output_size = 5  # 輸出類別的數量
dropout_rate = 0.1  # dropout 的比率

# 創建 ConformerModel
conformer_model = ConformerModel(input_size, d_model, n_head, feedforward_dim, num_layers, output_size, dropout_rate)

# 使用 Wav2Vec2 模型和相應的 tokenizer
wav2vec2_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")

# 將 ConformerModel 與 Wav2Vec2 模型結合
full_model = ConformerWithWav2Vec2(conformer_model, wav2vec2_model)

# 載入音訊文件
input_waveform, sample_rate = torchaudio.load("C:\\Users\\fclin\\Desktop\\test0125\\01252.wav", normalize=True)

# 使用 Librosa 轉換為頻譜圖
spectrogram = librosa.feature.melspectrogram(y=input_waveform.numpy().flatten(), sr=sample_rate)

print("Spectrogram Shape:", spectrogram.shape)

# 取得特定時間範圍的頻譜向量
time_start = 0.0  # 開始時間（以秒為單位）
time_end = 2.0    # 結束時間（以秒為單位）

# 將時間轉換為樣本點索引
sample_start = int(time_start * sample_rate / librosa.get_duration(y=input_waveform.numpy().flatten(), sr=sample_rate))
sample_end = int(time_end * sample_rate / librosa.get_duration(y=input_waveform.numpy().flatten(), sr=sample_rate))

# 取得頻譜向量
spectrogram_vector = spectrogram[:, sample_start:sample_end]

# 檢查頻譜向量是否為空
if not np.any(spectrogram_vector):
    print("Empty Spectrogram Vector")
else:
    # 將頻譜向量取 log
    spectrogram_vector_log = np.log(spectrogram_vector + 1e-9)  # 避免 log(0)
    # 打印頻譜向量
    print("Spectrogram Vector (log):", spectrogram_vector_log)

# 將頻譜向量進行離散餘弦轉換（DCT）
mfcc = dct(spectrogram_vector_log, type=2, axis=0, norm='ortho')[:13]  # 取13個MFCC系數

# 打印MFCC的形狀
print("MFCC Shape:", mfcc.shape)

