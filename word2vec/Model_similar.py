
from gensim.models import word2vec

model = word2vec.Word2Vec.load('/Users/riku/Downloads/プロジェクト/自学/model/word2vec.model')

# 結果の確認
# 関数most_similarを使って「」の類似単語を調べる。
ret = model.wv.most_similar(positive=['魔女'])
for item in ret:
    print(item[0], "%.3f" % item[1])

#out = model.most_similar(positive=[u"エミリア", u"主人公"], negative=[u"ヒロイン"])
#for x in out:
#    print(x[0])
