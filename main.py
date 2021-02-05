"""
run Word2Vec
"""
import argparse

from lib.util import Config
from lib.word2vec import CbowModel, SkipGramModel

def main():
    # load configs
    dconf_path = 'config/data.json'
    mconf_path = 'config/word2vec.json'
    dconf = Config(dconf_path)
    mconf = Config(mconf_path)

    # load w2v model and train
    if mconf.model == 'cbow':
        w2v = CbowModel(dconf, mconf)
    else:
        w2v = SkipGramModel(dconf, mconf)

    w2v.train()
    w2v.save('trained.pth')

    # test w2v
    word = 'hospital'
    print(w2v.nearest(word))

    print(w2v.similarity(word, 'attacks').item())
    print(w2v.similarity(word, word).item())


if __name__ == '__main__':
    main()
