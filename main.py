"""
run Word2Vec
"""
import argparse

from lib.util import Config
from lib.word2vec import CbowModel, SkipGramModel


def main(args):
    # load configs
    dconf_path = 'config/data.json'
    mconf_path = 'config/word2vec.json'
    dconf = Config(dconf_path)
    mconf = Config(mconf_path)
    # load w2v model and train
    if mconf.model == 'cbow':
        w2v = CbowModel(dconf, mconf, args.mode)
    else:
        w2v = SkipGramModel(dconf, mconf, args.mode)

    if args.mode != 'test':
        w2v.train()
        w2v.save(dconf.saved_file)

    # test w2v
    word = 'hospital'
    print(w2v.nearest(word))
    print(w2v.similarity(word, 'attacks').item())
    print(w2v.similarity(word, word).item())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transformer Neural Machine Translation')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'retrain'])
    args = parser.parse_args()
    main(args)
