from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import streamlit as st
import re
import nltk
nltk.download('vader_lexicon')
from nltk.metrics import edit_distance

#my list of branches/departments/agency
word_list=[
    "ABERGAVENNY",
    "ABERTILLERY",
    "BLACKOOD",
    "CLEVEDON",
    "GRIFFITHSTOWN",
    "HEREFORD",
    "KENFIG HILL",
    "PENARTH",
    "PORTISHEAD",
    "USK",
    "ROSS ON WYE",
    "WHITCHURCH",
    "NEWPORT",
    "CAERLEON ROAD",
    "SWANSEA",
    "CWMBRAN",
    "HANDPOST",
    "23",
    "63",
    "34",
    "58",
    "RISCA",
    "HEAD OFFICE",
    "CALDICOT",
    "MONMOUTH",
    "CHEPSTOW",
    "HO",
    "BIRCHGROVE",
    "BRECON",
    "CARDIFF",
    "ONLINE",
    "POSTAL",
    "SAVINGS & CUSTOMER CONTACT",
    "DIRECT MORTGAGE SALES",
    "DIRECT SALES",
    "LENDING OPERATIONS",
    "BUSINESS DEVELOPMENT",
    "BANKING HALL",
    "BDA",
    "BRANCHES",
    "BROKER DEVELOPMENT",
    "COMMUNICATIONS",
    "CREDIT CONTROL",
    "CSS",
    "FASTER PAYMENTS",
    "MY ACCOUNTS",
    "PRODUCTS",
    "SCC",
    "QA",
    "Mortgage servicing"
]

analyzer = SentimentIntensityAnalyzer()

st.header("Sentiment Analysis")

#analyse text
with st.expander("Analyze Text"):
    text = st.text_input('Text here: ')
    if text:
        sentiment_scores = analyzer.polarity_scores(text)
        polarity = sentiment_scores['compound']
        sentiment_label = 'Positive' if polarity > 0 else ('Negative' if polarity < 0 else 'Neutral')
        st.write('Polarity:', round(polarity, 2))
        st.write('Sentiment:', sentiment_label)

# with csv

with st.expander("Analyze csv"):
    upl = st.file_uploader('Upload csv file')

    def score(x):
        sentiment_scores = analyzer.polarity_scores(x)
        return sentiment_scores['compound']

    def analyse(x):
        polarity = x
        return 'Positive' if polarity > 0 else ('Negative' if polarity < 0 else 'Neutral')

        #interest word is coming for the rate and hence it is not a +ve word so replacing this word with just rate
        #as this interest word is causing problem
    def replace_interest_rate(text):
        return re.sub(r'\binterest \b', 'rate', text)

    if upl:
        df = pd.read_csv(upl)

        df['Comments1'] = df['Comments'].apply(replace_interest_rate)

        df['score'] = df['Comments1'].apply(score)
        df['Sentiment'] = df['score'].apply(analyse)

        #get positive words if sentiment is positive, negative words if sentiment is neg, otherwise []
        def get_words(text, sentiment):
            words = text.split()
            if sentiment == 'Positive':
                words = [word for word in words if analyzer.polarity_scores(word)['compound'] > 0]
                return words
            elif sentiment == 'Negative':
                words = [word for word in words if analyzer.polarity_scores(word)['compound'] < 0]
                return words
            else:
                words = [word for word in words if analyzer.polarity_scores(word)['compound'] == 0]
                return words

        df['main_words']=df.apply(lambda row: get_words(row['Comments1'], row['Sentiment']), axis=1)

        #get similar words-for branch/agency list
        def find_similar_word(phrase,phrase_list):
            if pd.isnull(phrase):
                return phrase

            min_distance = float('inf')
            closest_phrase = phrase

            for ph in phrase_list:
                distance = edit_distance(phrase.lower(), ph.lower())
                if distance < min_distance:
                    min_distance = distance
                    closest_phrase = ph
            if min_distance >= len(phrase):
                return phrase
            else:
                return closest_phrase

        df['branch_ag_dept'] = df['Branch/Agency/Department'].apply(lambda x: find_similar_word(x,word_list))

        def replace_specific_words(phrase):
            if isinstance(phrase, str) and ("newport" in phrase.lower() or "ho" in phrase.lower() or "head office" in phrase.lower()):
                return "Head Office"
            return phrase

        df['branch_ag_dept'] = df['branch_ag_dept'].apply(replace_specific_words)
        df.drop('Comments1',axis=1,inplace=True)
        st.write(df.head(10))

        @st.cache_data
        def convert_df(df):
            return df.to_csv(index=False).encode('utf-8')

        csv = convert_df(df)

        st.download_button(label='Download csv', data=csv, file_name='sentiment.csv', mime='text/csv')
