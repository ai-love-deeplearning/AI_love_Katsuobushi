
from gensim.models import word2vec
import os

class WordEmbedding(object):

    MODEL_FILE = '/Users/riku/Downloads/プロジェクト/自学/model/word2vec.model'

    def __init__(self):
        self.model = None

    def load(self, filename=MODEL_FILE):
        """ Load word2vec model"""
        if not os.path.exists(filename):
            raise Exception('Model not found: {}'.format(filename))
        self.model = word2vec.Word2Vec.load(filename)

    def save(self, filename=MODEL_FILE):
        self.model.save(filename)

    def get_model(self):
        return self.model

    def get_vocabulary_size(self):
        return len(self.model.wv.vocab)

    def get_most_similar(self, positive=[], negative=[]):
        return self.model.wv.most_similar(positive=positive, negative=negative)

    def save_embedding_projector_files(self, vector_file, metadata_file):
        with open(vector_file, 'w', encoding='utf-8') as f, open(metadata_file, 'w', encoding='utf-8') as g:
            # metadata file needs header
            g.write('Word\n')

            for word in self.model.wv.vocab.keys():
                embedding = self.model.wv[word]

                # Save vector TSV file
                f.write('\t'.join([('%f' % x) for x in embedding]) + '\n')

                # Save metadata TSV file
                g.write(word + '\n')

def embedding_projector():
    embedding = WordEmbedding()
    embedding.load()

    print('Vocaburary size:', embedding.get_vocabulary_size())
    embedding.save_embedding_projector_files('vector.tsv', 'metadata.tsv')

if __name__ == '__main__':
    embedding_projector()
