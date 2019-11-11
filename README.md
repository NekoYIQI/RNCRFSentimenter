# RNCRFSentimenter
Reproduction of in proceedings Recursive Neural Conditional Random Fields for Aspect-based Sentiment Analysis. Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing, 616-626. Wang, W., Pan, S., Dahlmeier, D., & Xiao, X. (2016). 

Dependencies:
```
-Python: 3.6.8
-Jupyter notebook: 5.7.4
-Stanford CoreNLP: 3.9.2
-genism.word2vec
-numpy: 1.16.2
-tensorflow: 1.12.0
-sklearn_crfsuite: 0.3
-joblib: 0.13.2
-Flask: 0.10.1
-beautifulsoup4: 4.7.1
-urllib3: 1.24.1
```
Start up procedures: 
Terminal:
```
java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000
```
cd to app:
```
python app.py
```
