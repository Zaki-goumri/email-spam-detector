import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")


def load_lkaggle_dataset():
    try:
        df = pd.read_csv("./dataset/spam.csv", encoding="latin-1")
        df = df[["v1", "v2"]].copy
        df.columns = ["label", "message"]
        df["label"].map({"spam": "spam", "ham": "ham"})
        print("✅ Successfully loaded Kaggle dataset!")
        return df["message"], df["label"]
    except FileNotFoundError:
        print("❌ Kaggle dataset not found. Using enhanced built-in dataset...")
        return load_enhanced_builtin_dataset()


def load_enhanced_builtin_dataset():
    emails_data = [
        # More diverse SPAM examples
        ("Get rich quick! Click here now!", "spam"),
        ("Congratulations! You've won $1000000!", "spam"),
        ("FREE MONEY! NO STRINGS ATTACHED!", "spam"),
        ("Act now! Limited time offer!", "spam"),
        ("URGENT: Claim your prize now!", "spam"),
        ("Make money from home easily!", "spam"),
        ("Nigerian prince needs your help", "spam"),
        ("Click here for amazing deals!", "spam"),
        ("Lose weight fast with this pill!", "spam"),
        ("You are pre-approved for credit!", "spam"),
        ("WIN BIG! Play now!", "spam"),
        ("FREE iPhone! Click immediately!", "spam"),
        ("URGENT ACTION REQUIRED!", "spam"),
        ("Congratulations winner! Claim now!", "spam"),
        ("Make $5000 per week working from home", "spam"),
        ("FINAL NOTICE: Your warranty expires today", "spam"),
        ("Click here to unsubscribe immediately", "spam"),
        ("You have been selected for a special offer", "spam"),
        ("Limited time: 90% discount on everything", "spam"),
        ("BREAKING: This stock will make you rich", "spam"),
        ("Your computer may be infected! Download now", "spam"),
        ("Meet singles in your area tonight", "spam"),
        ("Congratulations! You qualify for instant loan", "spam"),
        ("ALERT: Suspicious activity on your account", "spam"),
        ("Win a free vacation! Enter now!", "spam"),
        ("Lose 30 pounds in 30 days guaranteed", "spam"),
        ("Your order is ready for pickup", "spam"),  # tricky spam
        ("CONGRATULATIONS LOTTERY WINNER!", "spam"),
        ("Free trial - no credit card required!", "spam"),
        ("Make money with cryptocurrency secrets", "spam"),
        # More diverse LEGITIMATE examples
        ("Meeting scheduled for tomorrow at 3pm", "ham"),
        ("Can you review the project proposal?", "ham"),
        ("Please find the attached document", "ham"),
        ("How was your weekend?", "ham"),
        ("The quarterly report is ready", "ham"),
        ("Let's grab coffee tomorrow", "ham"),
        ("Project deadline is next Friday", "ham"),
        ("Happy birthday! Hope you have a great day", "ham"),
        ("Can you send me the budget numbers?", "ham"),
        ("Great job on the presentation", "ham"),
        ("Thanks for your help yesterday", "ham"),
        ("See you at the conference next week", "ham"),
        ("Please confirm your attendance", "ham"),
        ("The meeting has been moved to room 205", "ham"),
        ("Can we reschedule our call?", "ham"),
        ("Here are the files you requested", "ham"),
        ("Looking forward to working with you", "ham"),
        ("Please review and provide feedback", "ham"),
        ("The client approved the proposal", "ham"),
        ("Team lunch is at noon today", "ham"),
        ("Your password has been updated successfully", "ham"),
        ("Welcome to our new employee portal", "ham"),
        ("Monthly newsletter - February edition", "ham"),
        ("Reminder: Staff meeting tomorrow at 2pm", "ham"),
        ("Your order has been shipped", "ham"),
        ("Thank you for your purchase", "ham"),
        ("Account statement is now available", "ham"),
        ("Class schedule for next semester", "ham"),
        ("Library books are due next week", "ham"),
        ("Your appointment is confirmed for Tuesday", "ham"),
    ]
    df = pd.DataFrame(emails_data, columns=["message", "label"])
    return df["message"], df["label"]


print("🤖 BUILDING MY FIRST AI - SPAM DETECTOR")
print("=" * 50)

X, Y = load_lkaggle_dataset()

print("📊 Dataset Analysis:")
print(f"Dataset size: {len(X):,} messages")
spam_ratio = (Y == "spam").mean()
print(f"   Spam ratio: {spam_ratio:.1%}")


X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42, stratify=Y
)

print(f"📚 Training set: {len(X_train)} emails")
print(f"🧪 Testing set: {len(X_test)} emails")
print()


models = {
    "Naive Bayes": Pipeline(
        [
            (
                "vectorizer",
                TfidfVectorizer(
                    stop_words="english",
                    lowercase=True,
                    max_features=5000,
                    ngram_range=(1, 2),
                ),
            ),
            ("classifier", MultinomialNB(alpha=0.1)),
        ]
    )
}

best_model = None
best_name = None
best_accuracy = 0
results = {}

for name, model in models.items():
    print(f"Training {name}...")

    model.fit(X_train, Y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(Y_test, y_pred)

    results[name] = {"model": model, "accuracy": accuracy, "predictions": y_pred}

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model
        best_name = name

print(f"🏆 Best Model: {best_name} ({best_accuracy:.1%} accuracy)")
print(classification_report(Y_test, results[best_name]["predictions"]))

test_emails = [
    "URGENT! Your account will be suspended!",
    "Can we schedule a meeting for next week?",
    "FREE GIFT! Limited time only!",
    "Thanks for sending the report",
    "You've won a million dollars!",  # This should be spam now!
    "Let's discuss the project tomorrow",
]

for email in test_emails:
    prediction = best_model.predict([email])[0]
    probabilities = best_model.predict_proba([email])[0]
    confidence = max(probabilities)

    status = "🚨 SPAM" if prediction == "spam" else "✅ LEGITIMATE"
    print(f"{status} ({confidence:.1%}): '{email}'")

print("\n" + "=" * 50)
print("🎉 IMPROVEMENTS MADE:")
print("✅ 3x larger dataset (60 vs 24 emails)")
print("✅ Better text processing (bigrams, max features)")
print("✅ Balanced dataset (equal spam/ham)")
print("✅ Stratified sampling")
print("✅ Tuned algorithm parameters")
