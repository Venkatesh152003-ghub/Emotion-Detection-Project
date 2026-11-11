# 🎯 Emotion Detection System using Hybrid Model

## 📌 Project Description
This project is a web-based Emotion Detection System that analyzes user input text and predicts the corresponding emotion. The system uses a hybrid approach combining Machine Learning (TF-IDF + SVM) and rule-based techniques to improve prediction accuracy.

---

## 🚀 Features
- Detects emotions from user text input
- Supports emotions:
  - Joy 😊
  - Sadness 😢
  - Anger 😡
  - Fear 😨
  - Disgust 🤢
  - Shame 😳
  - Guilt 😔
- Web interface using Flask
- Improved accuracy using keyword-based rules

---

## 🛠️ Technologies Used
- Python
- Flask
- Scikit-learn
- NumPy
- Joblib
- HTML, CSS

---

## 📁 Project Structure
project_folder/
│
├── app.py
├── hybrid_model.pkl
├── tokenizer.pkl
├── tfidf_vectorizer.pkl
│
├── templates/
│ └── index.html
│
├── static/
│ └── style.css

---

## ⚙️ Installation & Setup

1. Clone the repository:
```bash
git clone https://github.com/your-username/emotion-detection.git
cd emotion-detection
2.Install dependencies:
pip install flask numpy scikit-learn joblib
3.Run the application:
python app.py
4.Open in browser:
http://127.0.0.1:5000/

## How It Works
1.User enters text in the web interface
2.Rule-based system checks for known emotional keywords
3.If no match → TF-IDF + ML model predicts emotion
4.Output is displayed on the same webpage
## Example Inputs
      Input	                       Output
I am very happy today	               Joy
I feel very sad	                    Sadness
I am frustrated with your service	   Anger
I am scared about exams            	 Fear

## Applications
1.Customer feedback analysis
2.Social media sentiment analysis
3.E-commerce review analysis
4.Mental health monitoring

## Future Enhancements
1.Improve model accuracy using deep learning
2.Add more emotions
3.Deploy on cloud (Heroku / AWS)
4.Mobile app integration


