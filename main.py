import json_lines
import pandas as pd
import time
import methods
import numpy as np
import pandas_profiling
from pandas_profiling import ProfileReport

from nltk.corpus import stopwords

from sklearn.decomposition import LatentDirichletAllocation

# Stop placed at global scope - why?
stop = set(stopwords.words("english"))

start = time.time()

class FileReader:

    def __init__(self, file_path, counter):
        with open(file_path) as f:
            self.body_text = []
            self.published_at = []
            self.counter = counter
            content = json_lines.reader(f)
            for c in content:
                if self.counter > 0:
                    # if self.counter % 100 == 0:
                    # print(self.counter)
                    self.body_text.append(c['body'])
                    self.published_at.append(c['published_at'])
                    self.counter -= 1
                else:
                    break

    def __repr__(self):
        return f'{self.counter}:{self.body_text}{self.published_at}'


# start main()_________________________________
def main():
    # Load the data from source into a dataframe

    article_num = 10
    file_path = "/Users/darrenking/Desktop/CT5157 - Data Mining/Week 1/data/aylien-covid-news.jsonl"
    content = FileReader(file_path, article_num)

    df = pd.DataFrame({"body": content.body_text, "date": content.published_at})
    # This is the main dataframe we're working with but for the purposes of this assignment I want to just deal with
    # the text data, I don't need the date data right now; see own work below.

    df = df["body"]

    print(df)

    all_rows = []
    for row in df.body:
        row = methods.create_vsm(row)
        all_rows.append(row)

    df['body_text'] = all_rows

    df.body_text = df.body_text.apply(methods.remove_stop_words)

    text = df.body_text.values # text is a numpy array, not a dataframe!
    X = methods.tf_idf_vectorise(text)

    # My own work outside the walk-through__________________________

    # Utilise a profile of the data to view it as a html page
    # profile = ProfileReport(df, explorative=True, dark_mode=True)
    # profile.to_file('dataframe_profile.html')

    # df = df.drop("date", axis=1)

    # df['clean_text'] = hero.clean(df['body']) - Unable to get the

    # methods.k_means_clustering_tsne(my_X, 5, df)
    # methods.k_means_clustering_umap(my_X, 5, df)

    # Sentiment analysis - As we're working with unlabelled data we'll use Snorkel to create labels




# end main()______________________________________

if __name__ == '__main__':
    main()

print("\n" + 50 * '#')
print(time.time() - start)
print(50 * '#' + '\n')
