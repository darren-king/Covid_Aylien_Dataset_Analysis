from sklearn.feature_extraction.text import TfidfVectorizer
import re
import string

from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords

# Stop placed at global scope - why?
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
    plt.savefig("Article_Clusters.png")
    plt.show()


def vectorise(text):
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
