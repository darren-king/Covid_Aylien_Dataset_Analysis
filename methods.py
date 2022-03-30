import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import string
import umap
from nltk.stem import SnowballStemmer
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords

stop = set(stopwords.words("english"))


def k_means_clustering_tsne(X, k, df):
    """
    This function uses the K-Means Clustering and T-Distributed Stochastic Neighbor Embedding APIs
    from the scikit-learn framework
    :param df: This is the dataframe we're working with
    :param X: This is our vectorised text parameter
    :param k: This is the number of clusters we want to use in
    :return:
    """

    kmeans = KMeans(n_clusters=k)
    y_pred = kmeans.fit_predict(X.toarray())
    df['y'] = y_pred  # just adding a column to our dataframe

    tsne = TSNE(verbose=1, perplexity=10)
    X_embedded = tsne.fit_transform(X.toarray())

    sns.set(rc={"figure.figsize": (15, 15)})

    palette = sns.hls_palette(len(set(y_pred)), l=0.4, s=0.9)

    # Plotting the data

    print("\nPlotting")
    sns.scatterplot(X_embedded[:, 0], X_embedded[:, 1], hue=y_pred, legend='full', palette=palette)
    plt.title("TSNE with KMeans Labels")
    plt.savefig("Article_Clusters_Tsne.png")
    plt.show()


def k_means_clustering_umap(X, k, df):
    """
    This function uses hte K-Means Clustering algorithm with the UMAP embedding / dimensionality
    reduction technique
    :param X: This is our vectorised text parameter
    :param k: This is the number of clusters we want to use
    :param df: This is the dataframe we're working with
    :return:
    """

    kmeans = KMeans(n_clusters=k)
    y_pred = kmeans.fit_predict(X.toarray())
    df['y'] = y_pred  # just adding a column to our dataframe

    reducer = umap.UMAP(metric='hellinger')
    X_embedded = reducer.fit_transform(X)

    sns.set(rc={"figure.figsize": (15, 15)})

    palette = sns.hls_palette(len(set(y_pred)), l=0.4, s=0.9)

    # Plotting the data

    print("\nPlotting")
    sns.scatterplot(X_embedded[:, 0], X_embedded[:, 1], hue=y_pred, legend='full', palette=palette)
    plt.title("Umap with KMeans Labels")
    plt.savefig("Article_Clusters_Umap.png")
    plt.show()


def tf_idf_vectorise(text):
    """
    This function converts a collection of raw text documents to a matrix of TF-IDF features
    :param text:
    :return: A TF-IDF matrix vector
    """

    # initialise vectoriser
    vectoriser = TfidfVectorizer()
    # cast data to tfidf representation
    X = vectoriser.fit_transform(text)
    return X


def create_vsm(doc):
    """
    This function takes in a text document and removes any punctuation and symbols from the text document
    :param doc:
    :return: document without punctuation
    """

    remove_symbols = string.punctuation
    pattern = r"[{}]".format(remove_symbols)
    doc = re.sub(pattern, " ", doc.strip().lower())
    doc = doc.split()
    return doc


def tokenize(doc):
    """
    This function divides the sentences of the document into words
    :param doc:
    :return: document in words
    """

    doc = doc.replace("\n", " ")
    words = nltk.tokenize.word_tokenize(doc)
    return words


def remove_stop_words(row):
    """
    This function removes stop words from each row fed into the function.
    It does this using the English stopwords from the Natural Language Toolkit
    :param row:
    :return: row without stopwords
    """

    row = [word for word in row if word not in stop]
    row = " ".join(row)
    return row


def stem(words):
    """
    This function will tokenize the sentence and stem the words in the sentence
    :param words: The words that need to be stemmed
    :return: stems
    """

    # stemmed_words = [stemmer.stem(word) for word in words]
    #return stemmed_words


def remove_punctuation(string):
    punc = '''!()-[]{};:'"\,<>./?@#$%&*^'''
    for symbol in string:
        if symbol in punc:
            string = string.replace(symbol, "")
        return string