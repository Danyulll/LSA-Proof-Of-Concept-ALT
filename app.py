import re
import numpy as np
import pandas as pd
import seaborn as sns
from io import BytesIO
from flask import Flask, render_template, request
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer
from nltk import FreqDist, bigrams
from collections import Counter
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.preprocessing import Normalizer
import base64
from matplotlib.figure import Figure


######## lsaDocSim Code ##################
# The idea to do this came from Efficient Recommendation System Using
# Latent Semantic Analysis

# Combine our stop words with default
stop_words = set(stopwords.words("english"))
custom_stop_words = {'ii','i','e','g', 'official', 'calendar', 'first','year','second','third','fourth','fall','spring','summer','winter','credit','granted',}
stop_words = stop_words.union(custom_stop_words)

# Function for tokenizing and cleaning documents for LSA model
def nltk_pipeline(input_string):
    # Tokenize input string and remove punctuation
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(input_string)
    # Remove tokens containing numbers
    tokens = [token for token in tokens if not re.search(r'\d', token)]
    # Convert tokens to lowercase and remove stopwords
    filtered_tokens = [token.lower() for token in tokens if token.lower() not in stop_words]
     # Perform stemming on filtered tokens
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in filtered_tokens]
    bigram_freq = FreqDist(bigrams(stemmed_tokens))
    frequent_bigrams = [bigram for bigram, freq in bigram_freq.items() if freq >= 2]
    final_tokens = stemmed_tokens + ['_'.join(bigram) for bigram in frequent_bigrams]
    token_counts = Counter(final_tokens)
    filtered_final_tokens = [token.encode('ascii', 'ignore').decode('ascii') for token in final_tokens if token_counts[token] > 0] # play with this number
    filtered_final_tokens
    return " ".join(filtered_final_tokens)

# Function for returning document similarity
# Note currently will retrain LSA model each query. Very inefficient but done because constantly updating and experimenting. Change this eventually.
def lsaDocSim(query_course,year,use_tfidf):
    df = pd.read_csv(r"./data/UBCO_Course_Calendar.csv").dropna(subset=['Course Description']).reset_index()

    # Clean data
    clean_text = []
    for i in range(0,df.shape[0]):
        clean_text.append(nltk_pipeline(df["Course Description"][i]))

    vectorizer = CountVectorizer(min_df=1)
    dtm = vectorizer.fit_transform(clean_text)

    tfidf_transformer = TfidfTransformer()
    tfidf_dtm = tfidf_transformer.fit_transform(dtm)

    if(use_tfidf == 'y'):
        # Fit LSA
        lsa = TruncatedSVD(4, algorithm='randomized')
        dtm_lsa = lsa.fit_transform(tfidf_dtm)
        dtm_lsa = Normalizer(copy=False).fit_transform(dtm_lsa)
    else:
        # Fit LSA
        lsa = TruncatedSVD(6, algorithm='randomized')
        dtm_lsa = lsa.fit_transform(dtm)
        dtm_lsa = Normalizer(copy=False).fit_transform(dtm_lsa)

    # Get document embeddings
    doc_embeddings = np.asmatrix(pd.DataFrame(dtm_lsa))

    # Get domain and year information
    other_info = np.asmatrix(df.iloc[:,7:])

    # Normalize the rows of the other info matrix
    row_norms = np.linalg.norm(other_info, axis=1)
    normalized_matrix = other_info / row_norms[:, np.newaxis]

    # Create final embedding
    mat = np.concatenate((normalized_matrix, doc_embeddings), axis=1)

    # Get course similarities
    doc_sim=mat*mat.T

    # Get mapping of sim matrix index to Course Code
    df['ID'] = df.index
    map = df.loc[:,['Course Code','ID']]
    map = map.set_index('Course Code').to_dict(orient='dict')['ID']

    # Get sim matrix index of query course
    index = map[query_course]

    # return top 3 recommendations and their scores filtered by year

    sorted_indices = np.argsort(doc_sim[index,:])

    sorted_values = doc_sim[index,sorted_indices]
    sorted_values = np.asarray(sorted_values)[0]
    sorted_values = sorted_values / max(sorted_values) # make largest sim 1

    rev_map = {value: key for key, value in map.items()}
    sorted_indices = list(np.asarray(sorted_indices)[0])[::-1]

    column_data = {
        "Course Code": [rev_map[idx] for idx in sorted_indices],
        "Course Name": df.iloc[sorted_indices,2],
        "Similarity": list(sorted_values)[::-1]
    }

    df_out = pd.DataFrame(column_data)
    df_out["Numeric Component"] = df_out["Course Code"].apply(lambda x: x.split()[1][0])

    if( int(year) <= 4 and int(year) >= 1):
        df_out = df_out[df_out["Numeric Component"] == year]
    
    return df_out.iloc[:,:3]

####### Flask Code ##############

app = Flask(__name__)

@app.route('/result', methods=['POST'])
def result():
    user_string = request.form['user_string']
    year = request.form['year']
    tfidf = request.form['tfidf']
    df_out = lsaDocSim(user_string,year,tfidf)
    html_table = df_out.head(3).to_html()

    # Generate the figure **without using pyplot**.
    fig = Figure()
    ax = fig.subplots()

    sns.barplot(data=df_out, y="Course Code", x="Similarity", ax=ax)

    # Save it to a temporary buffer.
    buf = BytesIO()
    fig.savefig(buf, format="png")
    # Embed the result in the html output.
    data = base64.b64encode(buf.getbuffer()).decode("ascii")
    return render_template('result.html', user_string=user_string, year=year, tfidf=tfidf, graph=f"<img src='data:image/png;base64,{data}'/>",html_table=html_table)

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)