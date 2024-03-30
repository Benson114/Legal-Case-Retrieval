import json
import logging
import sys
from pathlib import Path

import yaml
from pyserini.search.lucene import LuceneSearcher

args = yaml.safe_load(open("config/config.yaml", "r", encoding="utf-8"))
STOPWORD_FILE_DIR = args["STOPWORD_FILE_DIR"]
BM25_TEXTS_DIR = args["BM25_TEXTS_DIR"]
BM25_TEXTS_INFO = args["BM25_TEXTS_INFO"]
BM25_TEXTS_INDEX_DIR = args["BM25_TEXTS_INDEX_DIR"]
SEARCH_THREADS = args["SEARCH_THREADS"]
RECALL_NUM_HITS = args["RECALL_NUM_HITS"]

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

from src.TextPreprocessing import BM25_TextPreProcessor
from src.SegIDParser import SegIDParser


def load_searcher():
    logger.info("Loading pyserini BM25 index.")
    searcher = LuceneSearcher(BM25_TEXTS_INDEX_DIR)
    logger.info("Loading done.")
    return searcher


def BM25_Recall(query, num_hits, searcher):
    """
    :param query: 查询文本/文档，是一个字符串
    :param num_hits: 初步召回结果的数量
    :param searcher: pyserini BM25检索器
    :return: 初步召回结果列表，每个元素是一个字典，包含id和score两个key
    """
    bm25_text_preprocessor = BM25_TextPreProcessor()
    bm25_text_preprocessor.get_stopwords(STOPWORD_FILE_DIR)

    tokenized_query = bm25_text_preprocessor.tokenize_text(query)
    length = len(tokenized_query)

    logger.info("Preliminary searching for relevant docs.")
    list_hits = []
    for i in range(0, length, 1024):
        hits = searcher.search(tokenized_query[i:i + 1024], k=RECALL_NUM_HITS)
        for hit in hits:
            list_hits.append(
                {
                    "id": hit.docid,  # hit.docid是pyserini检索结果的文档id，在这里指的是doc_id
                    "score": hit.score
                }
            )
    logger.info(f"Preliminary searching done. [Num of docs: {len(list_hits)}]")

    logger.info("Fetching original recalling results.")
    list_hits.sort(key=lambda x: x["score"], reverse=True)
    list_hits_deduplicated = []
    for hit in list_hits:
        if hit not in list_hits_deduplicated:
            list_hits_deduplicated.append(hit)
        if len(list_hits_deduplicated) >= num_hits:
            break
    logger.info(f"Fetching done. [Num of original hits: {len(list_hits)}]")

    return list_hits_deduplicated[:num_hits]


if __name__ == "__main__":
    lecardv2_recall = {}
    searcher = load_searcher()
    searcher.num_threads = SEARCH_THREADS if SEARCH_THREADS else 1
    with open("eval/lecardv2_all_query.json", "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            logger.info(f"Recalling for query {idx + 1}.")
            line = json.loads(line)
            q_id = line["id"]
            q_text = line["query"]

            hits = BM25_Recall(q_text, RECALL_NUM_HITS, searcher)
            lecardv2_recall[q_id] = [hit["id"] for hit in hits]

            with open("lecardv2_all_recall.json", "w", encoding="utf-8") as x:
                json.dump(lecardv2_recall, x, ensure_ascii=False)
