# 必要なライブラリのインポート
from pyannote.audio import Pipeline
import whisper
import numpy as np
from pydub import AudioSegment

# 話者分離モデルの初期化
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")

# Whisperモデルのロード
model = whisper.load_model("large-v3")

# 音声ファイルを指定
audio_file = "JA_B00000_S00529_W000007.mp3"  # MP3ファイルを指定します

# 話者分離の実行
diarization = pipeline(audio_file)

# MP3ファイルをAudioSegmentで読み込む
audio_segment = AudioSegment.from_file(audio_file, format="mp3")

# 音声ファイルを16kHz、モノラルに変換
audio_segment = audio_segment.set_frame_rate(16000).set_channels(1)

# 話者分離の結果をループ処理
for segment, _, speaker in diarization.itertracks(yield_label=True):
    # 話者ごとの発話区間の音声を切り出し（ミリ秒単位）
    start_ms = int(segment.start * 1000)
    end_ms = int(segment.end * 1000)
    segment_audio = audio_segment[start_ms:end_ms]

    # 音声データをnumpy配列に変換
    waveform = np.array(segment_audio.get_array_of_samples()).astype(np.float32)

    # 音声データを[-1.0, 1.0]の範囲に正規化
    waveform = waveform / np.iinfo(segment_audio.array_type).max

    # Whisperによる文字起こし
    # 音声データをサンプリングレート16kHzに合わせて、テンソルに変換
    result = model.transcribe(waveform,fp16=False)

    # 話者ラベル付きで結果をフォーマットして出力
    for data in result["segments"]:
        start_time = segment.start + data["start"]
        end_time = segment.start + data["end"]
        print(f"{start_time:.2f},{end_time:.2f},{speaker},{data['text']}")