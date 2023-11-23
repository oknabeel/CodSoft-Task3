import streamlit as st
import nltk
import pickle
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer
nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()

# ==============================================================================
def transform_text(text):
  text = text.lower()
  text = nltk.word_tokenize(text)

  new = []
  for i in text:
    if i.isalnum():
      new.append(i)

  text = new[:]
  new.clear()

  for i in text:
    if i not in stopwords.words('english') and i not in string.punctuation:
      new.append(i)

  text = new[:]
  new.clear()

  for i in text:
    new.append(ps.stem(i))

  return " ".join(new)
# ====================================================================================

tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title('SMS/Email Spam Classifier')

input_sms = st.text_area('Enter the Message')

if st.button('Predict'):

    # Pre-Process
    transformed_sms = transform_text(input_sms)

    # Vectorize
    vector_input = tfidf.transform([transformed_sms])

    # Predict
    result = model.predict(vector_input)

    # Display
    if result == 1:
        st.header('Spam')
    else:
        st.header('Not Spam')