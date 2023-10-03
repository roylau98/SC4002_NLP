import gensim.downloader


def main():
    word2vec_google_news = gensim.downloader.load('word2vec-google-news-300')
    print(f"Vector for student: {word2vec_google_news['student']}", end="\n")
    print(f"Vector for Apple: {word2vec_google_news['Apple']}", end="\n")
    print(f"Vector for apple: {word2vec_google_news['apple']}", end="\n")


if __name__ == '__main__':
    main()
