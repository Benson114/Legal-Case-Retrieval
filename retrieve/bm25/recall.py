import json
import sys, logging

from tqdm import tqdm
from pathlib import Path
from pyserini.search.lucene import LuceneSearcher

import yaml

args = yaml.safe_load(open("config/config.yaml", "r", encoding="utf-8"))
STOPWORD_FILE_DIR = args["STOPWORD_FILE_DIR"]
BM25_SEGS_DIR = args["BM25_SEGS_DIR"]
BM25_SEGS_INFO = args["BM25_SEGS_INFO"]
BM25_SEGS_INDEX_DIR = args["BM25_SEGS_INDEX_DIR"]
SEARCH_K = args["SEARCH_K"]
SEARCH_THREADS = args["SEARCH_THREADS"]
RECALL_NUM_HITS = args["RECALL_NUM_HITS"]

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

from src.TextPreprocessing import BM25_TextPreProcessor


def load_searcher():
    logger.info("Loading pyserini BM25 index.")
    searcher = LuceneSearcher(BM25_SEGS_INDEX_DIR)
    logger.info("Loading done.")
    return searcher


def BM25_Recall(query, num_hits, searcher):
    bm25_text_preprocessor = BM25_TextPreProcessor()
    bm25_text_preprocessor.get_stopwords(STOPWORD_FILE_DIR)

    logger.info(f"Splitting query into segments.")
    query_segs = bm25_text_preprocessor.doc2segs(query, 10, 5)
    logger.info(f"Splitting done. [Num of query segs: {len(query_segs)}]")

    logger.info("Preliminary searching for relevant segments of each query segments.")
    list_hits = []
    for seg in query_segs:
        hits = searcher.search(seg[0:1024], k=SEARCH_K)
        for hit in hits:
            list_hits.append(
                {
                    "id": hit.docid,
                    "score": hit.score
                }
            )
    logger.info(f"Preliminary searching done. [Num of segments: {len(list_hits)}]")

    logger.info("Fetching original recalling results.")
    list_hits.sort(key=lambda x: x["score"], reverse=True)
    list_hits_dedup = []
    for hit in list_hits:
        doc_id = hit["id"].split("_segment_")[0]
        if doc_id not in list_hits_dedup:
            list_hits_dedup.append(doc_id)
        if len(list_hits_dedup) > num_hits:
            break
    logger.info(f"Fetching done. [Num of original hits: {len(list_hits)}]")

    return list_hits_dedup[:num_hits]


if __name__ == "__main__":
    lecardv2_recall = {}
    searcher = load_searcher()
    searcher.num_threads = SEARCH_THREADS if SEARCH_THREADS else 1
    with open("lecardv2_query.json", "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            logger.info(f"Recalling for query {idx + 1}.")
            line = json.loads(line)
            q_id = line["id"]
            q_text = line["query"]

            hits = BM25_Recall(q_text, RECALL_NUM_HITS, searcher)
            lecardv2_recall[str(q_id)] = [hit.split("_")[-1] for hit in hits]

            with open("lecardv2_recall.json", "w", encoding="utf-8") as x:
                json.dump(lecardv2_recall, x, ensure_ascii=False)
                x.close()
        f.close()
