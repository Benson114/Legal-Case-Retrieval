import os
import re, jieba
import json, jsonlines

from glob import glob


class BM25_TextPreProcessor:
    def __init__(self):
        self.stopwords = set()

    @staticmethod
    def remove_empty(list_):
        return list(filter(None, list_))

    @staticmethod
    def remove_stopword(list_, stopwords):
        return list(filter(lambda word: word not in stopwords, list_))

    def get_stopwords(self, STOPWORD_FILE_DIR):
        """
        Args:
            STOPWORD_FILE_DIR: str, 停用词文件路径, 可以是一批停用词文件所在文件夹的路径或一个停用词文件的路径, 文件格式需要为 txt
        """
        if STOPWORD_FILE_DIR.endswith(".txt"):
            with open(STOPWORD_FILE_DIR, "r", encoding="utf-8") as f:
                self.stopwords.update(f.read().splitlines())
        else:
            for file in glob(os.path.join(STOPWORD_FILE_DIR, "*.txt")):
                with open(file, "r", encoding="utf-8") as f:
                    self.stopwords.update(f.read().splitlines())

    def tokenize_text(self, text, rm_stopwords=True):
        """
        Args:
            text: str, 待分词的文本
            rm_stopwords: bool, 是否去除停用词
        """
        text = text.strip()
        words = jieba.cut(text, cut_all=False, HMM=True)
        words = self.remove_empty(words)
        if rm_stopwords:
            words = self.remove_stopword(words, self.stopwords)
        tokenized_text = " ".join(words)
        return tokenized_text

    def tokenize_docfile(self, docfile):
        """
        Args:
            docfile: str, 文档文件路径
        """
        with open(docfile, "r", encoding="utf-8") as f:
            doc = json.load(f)
        doc_id = doc["id"]
        doc_text = doc["text"]
        tokenized_text = self.tokenize_text(doc_text)
        doc_dict = {
            "id": doc_id,
            "contents": tokenized_text,
        }
        return doc_dict


class DPR_TextPreProcessor:
    def __init__(self):
        pass

    @staticmethod
    def remove_empty(list_):
        return list(filter(None, list_))

    def sentencize_text(self, doc, max_length):
        """
        Args:
            doc: str, 文档内容
            max_length: int, 每个分段的最大长度
        """
        delimiters = [
            "(?<=[。！!？?])",
            "(?<=[；;])",
            "(?<=[：:])",
            "(?<=[，,])",
            ""
        ]
        sents = [doc]
        for delimiter in delimiters:
            new_sents = []
            for sent in sents:
                if len(sent) < max_length / 4:
                    new_sents.append(sent)
                elif delimiter:
                    new_sents.extend(re.split(delimiter, sent))
            sents = new_sents
        sents = self.remove_empty(sents)
        return sents

    def doc2segs(self, doc, max_length, min_overlap):
        """
        Args:
            doc: str, 文档内容
            max_length: int, 每个分段的最大长度
            min_overlap: int, 分段之间的最小重叠长度
        """
        sents = self.sentencize_text(doc, max_length)
        segs = []
        i, n = 0, len(sents)
        while i < n:
            seg = ""
            while i < n and len(seg) + len(sents[i]) < max_length:
                seg += sents[i]
                i += 1
            segs.append(seg)
            if i < n:
                overlap, j = 0, i - 1
                while j > 0 and overlap < min_overlap:
                    overlap += len(sents[j])
                    j -= 1
                i = j + 1
        return segs

    def docfile2segs(self, docfile, max_length, min_overlap):
        """
        Args:
            docfile: str, 文档文件路径
            max_length: int, 每个分段的最大长度
            min_overlap: int, 分段之间的最小重叠长度
        """
        segs_with_id = []
        with open(docfile, "r", encoding="utf-8") as f:
            doc = json.load(f)
        doc_id = doc["id"]
        doc_text = doc["text"]
        segs = self.doc2segs(doc_text, max_length, min_overlap)
        for i, seg in enumerate(segs):
            seg_dict = {
                "id": f"{doc_id}_segment_{i}",
                "contents": seg,
            }
            segs_with_id.append(seg_dict)
        return segs_with_id
