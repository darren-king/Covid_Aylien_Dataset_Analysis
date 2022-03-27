import json_lines
import pandas as pd
import time
import methods

import nltk
from nltk.corpus import stopwords
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

    article_num = 100
    file_path = "/Users/darrenking/Desktop/CT5157 - Data Mining/Week 1/data/aylien-covid-news.jsonl"
    content = FileReader(file_path, article_num)

    df = pd.DataFrame({"body": content.body_text, "date": content.published_at})

    # Rather than using a for loop, could we iterate using vectorisation? - look this up

    all_rows = []
    for row in df.body:
        row = methods.create_vsm(row)
        all_rows.append(row)

    df["body_text"] = all_rows

    df.body_text = df.body_text.apply(methods.remove_stop_words)

    # The text is now clean

    text = df.body_text.values
    X = methods.vectorise(text)  # csr.matrix - TF-IDF matrix

# My own work outside the walk-through__________________________

# So I just want the text part of the dataframe please and thanks

    df = df.drop(["body", "date"], axis=1)
    # print(df.info())



# end main()______________________________________


if __name__ == '__main__':
    main()

print("\n" + 50 * '#')
print(time.time() - start)
print(50 * '#' + '\n')
