from datasets import load_dataset
import soundfile as sf
import os

os.makedirs("data/real", exist_ok=True)
os.makedirs("data/ai", exist_ok=True)

ds = load_dataset(
    "SpeechAntiSpoofingBenchmarks/ASVspoof2021_LA",
    split="test",
    streaming=True
)

real_count = 0
ai_count = 0

print("Collecting samples...")

for x in ds:
    label = ds.features["label"].int2str(x["label"])
    audio = x["audio"]

    if label == "bonafide" and real_count < 10:
        real_count += 1
        filename = f"data/real/real_{real_count:02d}.wav"
        sf.write(filename, audio["array"], audio["sampling_rate"])
        print(f"Real {real_count}/10 saved")

    elif label == "spoof" and ai_count < 10:
        ai_count += 1
        filename = f"data/ai/ai_{ai_count:02d}.wav"
        sf.write(filename, audio["array"], audio["sampling_rate"])
        print(f"AI {ai_count}/10 saved")

    if real_count >= 10 and ai_count >= 10:
        break

print()
print("DONE!")
print("Real samples:", real_count)
print("AI samples:", ai_count)