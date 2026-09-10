from transformers import pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import os

MODEL = "mo-thecreator/Deepfake-audio-detection"

print("Loading AI voice detection model...")
classifier = pipeline("audio-classification", model=MODEL)
print("Model loaded!\n")


def get_prediction(output):
    label = output[0]["label"].lower()

    if any(word in label for word in ["fake", "spoof", "ai", "synthetic", "generated"]):
        return "AI"

    if any(word in label for word in ["real", "bonafide", "genuine", "human"]):
        return "REAL"

    return "UNKNOWN"


y_true = []
y_pred = []

for folder, actual_label in [
    ("data/real", "REAL"),
    ("data/ai", "AI")
]:

    print("=" * 50)
    print(f"Testing {actual_label} samples")
    print("=" * 50)

    files = sorted([
        f for f in os.listdir(folder)
        if f.lower().endswith(".wav")
    ])

    for filename in files:
        filepath = os.path.join(folder, filename)

        try:
            output = classifier(filepath)

            prediction = get_prediction(output)
            confidence = output[0]["score"] * 100

            y_true.append(actual_label)
            y_pred.append(prediction)

            status = "CORRECT" if prediction == actual_label else "WRONG"

            print(
                f"{filename:15} | "
                f"Actual: {actual_label:4} | "
                f"Predicted: {prediction:7} | "
                f"{confidence:6.2f}% | "
                f"{status}"
            )

        except Exception as e:
            print(f"ERROR with {filename}: {e}")


valid = [
    (true, pred)
    for true, pred in zip(y_true, y_pred)
    if pred in ["REAL", "AI"]
]

y_true = [x[0] for x in valid]
y_pred = [x[1] for x in valid]


print("\n" + "=" * 50)
print("FINAL EVALUATION")
print("=" * 50)

if len(y_true) > 0:

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, pos_label="AI", zero_division=0)
    recall = recall_score(y_true, y_pred, pos_label="AI", zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label="AI", zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=["REAL", "AI"])

    print(f"Total Samples : {len(y_true)}")
    print(f"Accuracy      : {accuracy * 100:.2f}%")
    print(f"Precision     : {precision * 100:.2f}%")
    print(f"Recall        : {recall * 100:.2f}%")
    print(f"F1 Score      : {f1 * 100:.2f}%")

    print("\nConfusion Matrix")
    print("                 Predicted")
    print("              REAL    AI")
    print(f"Actual REAL   {cm[0][0]:4}   {cm[0][1]:4}")
    print(f"Actual AI     {cm[1][0]:4}   {cm[1][1]:4}")

else:
    print("No valid predictions found.")

print("=" * 50)
