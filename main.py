"""
run Word2Vec
"""
from lib.util import Config
from lib.word2vec import CbowModel, SkipGramModel


# load configs
dconf_path = 'config/data.json'
mconf_path = 'config/word2vec.json'
dconf = Config(dconf_path)
mconf = Config(mconf_path)

# load w2v model and train
w2v = CbowModel(dconf, mconf)
# w2v = SkipGramModel(dconf, mconf)

w2v.train()

# test w2v
word = 'economic'
print(w2v.nearest(word))

print(w2v.similarity(word, 'biden').item())
print(w2v.similarity(word, word).item())
