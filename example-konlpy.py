import re
import bm25s
import Stemmer
from typing import List, Union
from konlpy.tag import Okt
from tokenization import tokenize

# Create your corpus here
corpus = [
    "a cat is a feline and likes to purr",
    "a dog is the human's best friend and loves to play",
    "a bird is a beautiful animal that can fly",
    "a fish is a creature that lives in water and swims",
    "Human machine interface for lab abc computer applications",
    "A survey of user opinion of computer system response time",
    "The EPS user interface management system",
    "System and human system engineering testing of EPS",
    "Relation of user perceived response time to error measurement",
    "The generation of random binary unordered trees",
    "The intersection graph of paths in trees",
    "Graph minors IV Widths of trees and well quasi ordering",
    "Graph minors A survey",
    "고양이는 고양이이며 가르릉거리는 것을 좋아한다",
    "개는 인간의 가장 친한 친구이며 놀기를 좋아한다",
    "새는 날 수 있는 아름다운 동물이다",
    "물고기는 물속에 살며 헤엄치는 생물이다",
    "실험실용 컴퓨터 응용 프로그램을 위한 휴먼 머신 인터페이스",
    "컴퓨터 시스템 응답 시간에 대한 사용자 의견 조사",
    "EPS 사용자 인터페이스 관리 시스템",
    "EPS의 시스템 및 인간 시스템 엔지니어링 테스트",
    "사용자가 인지하는 응답 시간과 오류 측정의 관계",
    "무작위 이진 정렬되지 않은 트리의 생성",
    "트리에서 경로의 교차 그래프",
    "그래프 마이너 IV 나무의 폭과 준순서",
    "그래프 미성년자 설문 조사",
]


# optional: create a stemmer
class CommonStemmer:
    def __init__(self):
        self.ko_tokenizer = Okt()
        self.en_tokenizer = Stemmer.Stemmer("english").stemWords
        self.en_token_pattern = re.compile(r"(?u)\b\w\w+\b")

    def is_korean(self, text: str):
        for char in text:
            if (
                "\uac00" <= char <= "\ud7a3"
                or "\u1100" <= char <= "\u11ff"
                or "\u3130" <= char <= "\u318f"
            ):
                return True
        return False

    def spliter(self, texts: Union[List[str], str]):
        if not isinstance(texts, list):
            if self.is_korean(texts):
                tokens = self.ko_tokenizer.pos(texts, norm=True, stem=True)
                texts = list(map(lambda x: x[0], tokens))
            else:
                texts = self.en_token_pattern.findall(texts)
        else:
            for i, text in enumerate(texts):
                if self.is_korean(text):
                    tokens = self.ko_tokenizer.pos(text, norm=True, stem=True)
                    tokens = list(map(lambda x: x[0], tokens))
                else:
                    tokens = self.en_token_pattern.findall(text)
                texts[i] = tokens
        return texts

    def __call__(self, texts: Union[List[str], str]):
        if not isinstance(texts, list):
            if self.is_korean(texts):
                tokens = self.ko_tokenizer.pos(texts, norm=True, stem=True)
            else:
                tokens = self.en_tokenizer(texts)
            texts = " ".join(list(map(lambda x: x[0], tokens)))
        else:
            for i, text in enumerate(texts):
                if self.is_korean(text):
                    tokens = self.ko_tokenizer.pos(text, norm=True, stem=True)
                    tokens = " ".join(list(map(lambda x: x[0], tokens)))
                else:
                    tokens = self.en_tokenizer(text)
                    tokens = "".join(tokens)
                texts[i] = tokens
        return texts


stemmer = CommonStemmer()
with open("stopwords.txt", "r", encoding="utf-8") as f:
    stopwords = list(f.read().splitlines())

corpus = stemmer(corpus)
# Tokenize the corpus and only keep the ids (faster and saves memory)
corpus_tokens = tokenize(corpus, stopwords=stopwords, stemmer=stemmer)

# Create the BM25 model and index the corpus
retriever = bm25s.BM25()
retriever.index(corpus_tokens)

# Query the corpus
query = "물고기가 고양이처럼 가르릉거려요?"
query = stemmer(query)
query_tokens = tokenize(query, stemmer=stemmer)
print(query_tokens)
# Get top-k results as a tuple of (doc ids, scores). Both are arrays of shape (n_queries, k)
results, scores = retriever.retrieve(query_tokens, corpus=corpus, k=2)

for i in range(results.shape[1]):
    doc, score = results[0, i], scores[0, i]
    print(f"Rank {i+1} (score: {score:.2f}): {doc}")

# You can save the corpus along with the model
retriever.save("animal_index_bm25_konlpy", corpus=corpus)
