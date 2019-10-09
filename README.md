# Install
```
git clone git@github.com:yonezu-qwt/lexrank.git
cd lexrank
sh setup.sh
```

# Usage
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
