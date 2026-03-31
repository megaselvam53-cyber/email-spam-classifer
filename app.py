from flask import Flask, render_template, request
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score
from sklearn.model_selection import train_test_split

app = Flask(__name__)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = text.strip()
    return text

df = pd.read_csv("spam.csv", encoding="latin-1")
df = df[['v1', 'v2']]
df.columns = ['label', 'message']
df['label'] = df['label'].map({'ham': 0, 'spam': 1})
df['message'] = df['message'].astype(str).apply(clean_text)

cv = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
X = cv.fit_transform(df['message'])
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = SVC(kernel='linear', class_weight='balanced')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)

@app.route('/')
def home():
    return render_template(
        'index.html',
        show_result=False,
        prediction="",
        message_text="",
        accuracy=round(accuracy * 100, 2),
        precision=round(precision * 100, 2),
        warning=""
    )

@app.route('/predict', methods=['POST'])
def predict():
    message = request.form.get('message', '').strip()

    if message == "":
        return render_template(
            'index.html',
            show_result=False,
            prediction="",
            message_text="",
            accuracy=round(accuracy * 100, 2),
            precision=round(precision * 100, 2),
            warning="Please enter a message first."
        )

    cleaned_message = clean_text(message)
    data = cv.transform([cleaned_message])
    pred = model.predict(data)[0]

    result = "🚨 Spam Message" if pred == 1 else "✅ Safe Message"

    return render_template(
        'index.html',
        show_result=True,
        prediction=result,
        message_text=message,
        accuracy=round(accuracy * 100, 2),
        precision=round(precision * 100, 2),
        warning=""
    )

if __name__ == "__main__":
    app.run(debug=True)