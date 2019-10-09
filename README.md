LexRank: Graph-based Lexical Centrality as Salience in Text Summarization
Gunes Erkan, Dragomir R. Radev
Journal Of Artificial Intelligence Research, Volume 22, pages 457-479, 2004

# Install
```
git clone git@github.com:yonezu-qwt/lexrank.git
cd lexrank
sh setup.sh
```

# Usage
## Scraping
```
git checkout -b scraping origin/scraping
```
## Train
```
# tfidf model
python train.py tfidf

# doc2vec model
python train.py doc2vec
```

## Summarize
```
# tfidf model
python lexrank_summarization.py tfidf sentence mmr

# doc2vec model
python lexrank_summarization.py doc2vec sentence mmr
```
