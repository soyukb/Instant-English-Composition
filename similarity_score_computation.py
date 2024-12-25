# 必要なライブラリをインポート
from sentence_transformers import SentenceTransformer  # 文をベクトルに変換するライブラリ
from sklearn.metrics.pairwise import cosine_similarity  # コサイン類似度を計算するための関数

def calculate_similarity(input_sentence, correct_sentences, model_name='all-MiniLM-L6-v2'):
    """
    入力文と複数の正解文の意味的な類似度を計算する関数。
    
    Args:
        input_sentence (str): 類似度を計算する対象の入力文。
        correct_sentences (list of str): 類似度を比較する正解文のリスト。
        model_name (str): SentenceTransformerで使用するモデル名（デフォルトは'all-MiniLM-L6-v2'）。
    """
    try:
        # モデルのロード
        # 指定された名前のSentenceTransformerモデルをロードする
        model = SentenceTransformer(model_name)
    except Exception as e:
        # モデルロード時にエラーが発生した場合の例外処理
        print(f"モデルのロードに失敗しました: {e}")
        return

    try:
        # 入力文のエンコード
        # 入力文を埋め込みベクトル（数値表現）に変換
        input_embedding = model.encode(input_sentence)
    except Exception as e:
        # エンコード時にエラーが発生した場合の例外処理
        print(f"入力文のエンコードに失敗しました: {e}")
        return

    # 類似度を格納する辞書を初期化
    results = {}
    for correct_sentence in correct_sentences:
        try:
            # 正解文のエンコード
            correct_embedding = model.encode(correct_sentence)
            # 入力文と正解文のコサイン類似度を計算
            similarity = cosine_similarity([input_embedding], [correct_embedding])[0][0]
            # 類似度を辞書に保存（キー: 正解文、値: 類似度スコア）
            results[correct_sentence] = similarity
        except Exception as e:
            # 類似度計算時にエラーが発生した場合の例外処理
            print(f"類似度計算でエラーが発生しました: {e}")
    
    # 計算結果を出力
    for sentence, score in results.items():
        print(f"Input: '{input_sentence}'")  # 入力文を表示
        print(f"Correct: '{sentence}'")     # 比較対象の正解文を表示
        print(f"Semantic similarity score: {score:.2f}\n")  # 類似度スコアを小数点2桁まで表示


# 使用例
input_sentence = "I am studying how to create an AI."
correct_sentences = [
    "I am learning to develop AI.",
    "I'm studying how to create an AI.",
    "I'm learning to develop AI.",
    "I am learning to develop AI and DeepLearning."
    "I'm learning develop AI.",
    "I enjoy programming artificial intelligence.",
    "I like cooking dinner."
]
calculate_similarity(input_sentence, correct_sentences)
