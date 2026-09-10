from transformers import pipeline
import os

MODEL = "garystafford/wav2vec2-deepfake-voice-detector"

print("Loading model...")
pipe = pipeline("audio-classification", model=MODEL)

def predict(path):
    result = pipe(path)
    fake_score = next(x["score"] for x in result if x["label"].lower() == "fake")
    return "fake" if fake_score >= 0.5 else "real", fake_score

correct = 0
total = 0

for folder, expected in [("data/real", "real"), ("data/ai", "fake")]:
    for file in sorted(os.listdir(folder)):
        if file.endswith(".wav"):
            path = os.path.join(folder, file)
            prediction, score = predict(path)

            total += 1
            if prediction == expected:
                correct += 1

            print(f"{file:20} Expected: {expected:4} | Predicted: {prediction:4} | Fake: {score:.2%}")

print("\n==============================")
print(f"Correct: {correct}/{total}")
print(f"Accuracy: {correct/total:.2%}")
print("==============================")
