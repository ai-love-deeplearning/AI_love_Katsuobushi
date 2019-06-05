
import codecs
import re
from janome.tokenizer import Tokenizer
from gensim.models import word2vec

# ファイル読込み、内部表現化
f = codecs.open('/Users/riku/Downloads/プロジェクト/自学/text_file/novel.txt', "r", "utf8")
text = f.read()
f.close()

# ファイル整形
deleteWords = ['！',
               '!',
               '?'
               '？',
               '「',
               '」',
               '（',
               '）',
               '.',
               '《',
               '》',
               '――']

for word in deleteWords:
    text = text.replace(word, '')

# 固有名詞の置き換え
text = text.replace('ハジメ', '少年')
# 空行の削除
text = re.sub('\n\n', '\n', text)
text = re.sub('\r', '', text)

# Tokenneizerインスタンスの生成
t = Tokenizer()

# テキストを引数として、形態素解析の結果、名詞・動詞原型のみを配列で抽出する関数を定義
def extract_words(text):
    tokens = t.tokenize(text)
    return [token.base_form for token in tokens if token.part_of_speech.split(',')[0] in['名詞', '動詞', '形容詞']]

#  関数テスト
#ret = extract_words('教室のざわめきに、ハジメは意識が覚醒していくのを感じた。')
#for word in ret:
#    print(word)

# 全体のテキストを句点('。')で区切った配列にする。
sentences = text.split('。')
# それぞれの文章を単語リストに変換(処理に数分かかる)
word_list = [extract_words(sentence) for sentence in sentences]

#print(word_list)

# 結果の一部を確認
#for word in word_list[0]:
#    print(word)

# size: 圧縮次元数
# min_count: 出現頻度の低いものをカット
# window: 前後の単語を拾う際の窓の広さを決定
# iter: 機械学習の繰り返し回数(デフォルト:5)十分学習できていないときにこの値を調整する
# model.wv.most_similarの結果が1に近いものばかりで、model.dict['wv']のベクトル値が小さい値ばかりのときは、学習回数が少ないと考えられる。
# その場合、iterの値を大きくして、再度学習を行う。

model = word2vec.Word2Vec(word_list,
                          size=100,
                          min_count=5,
                          window=15,
                          iter=20)

# モデル保存
model.save('/Users/riku/Downloads/プロジェクト/自学/model/word2vec.model')

# 結果の確認
# 関数most_similarを使って「」の類似単語を調べる。
#ret = model.wv.most_similar(positive=['彼'])
#for item in ret:
#    print(item[0], "%.3f" % item[1])

#print(len(model.wv.vocab))
