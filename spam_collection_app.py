import streamlit as st
import joblib
import string
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords

# Sayfa Yapılandırması
st.set_page_config(
    page_title="Spam SMS Tespit Sistemi",
    page_icon="📩",
    layout="centered"
)

# Eğitilmiş modeli ve vektörleştiriciyi yükleme
@st.cache_resource
def load_nlp_tools():
    model = joblib.load("spam_model.pkl")
    vec = joblib.load("tfidf_vectorizer.pkl")
    return model, vec

model, vec = load_nlp_tools()
stemmer = SnowballStemmer("english")
stop_words = set(stopwords.words("english"))

def clean_text(message):
    message = message.translate(str.maketrans("", "", string.punctuation))
    words = [stemmer.stem(word) for word in message.split() if word.lower() not in stop_words]
    return " ".join(words)

st.title("📩 Spam SMS Tespit Sistemi (NLP)")
st.write("Bu uygulama, girdiğiniz mesajın metnini analiz ederek **Spam** olup olmadığını yapay zeka ile tespit eder.")

user_message = st.text_area("Bir mesaj metni girin:", "Congratulations! You've won a free ticket to Bahamas. Call now!")

if st.button("Mesajı Analiz Et"):
    if user_message.strip() != "":
        with st.spinner("Analiz yapılıyor..."):
            cleaned = clean_text(user_message)
            features = vec.transform([cleaned])
            prediction = model.predict(features)[0]
            probability = model.predict_proba(features)[0]

            st.subheader("Sonuç:")
            if prediction == 1:
                risk_score = probability[1] * 100
                st.error(f"🚨 **DİKKAT! Bu mesaj bir SPAM mesajıdır!** (Risk Oranı: %{risk_score:.2f})")
            else:
                ham_score = probability[0] * 100
                st.success(f"✅ **Bu mesaj Güvenli (Ham) görünmektedir.** (Güven Oranı: %{ham_score:.2f})")
    else:
        st.warning("Lütfen analiz için geçerli bir metin girin.")