import json
import logging
import sys
from pathlib import Path

import yaml
from pyserini.search.lucene import LuceneSearcher

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
from src.SegIDParser import SegIDParser


def load_searcher():
    logger.info("Loading pyserini BM25 index.")
    searcher = LuceneSearcher(BM25_SEGS_INDEX_DIR)
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
                    "seg_id": hit.docid,  # hit.docid是pyserini检索结果的文档id，在这里指的是SegID().seg_id
                    "score": hit.score
                }
            )
    logger.info(f"Preliminary searching done. [Num of segments: {len(list_hits)}]")

    logger.info("Fetching original recalling results.")
    list_hits.sort(key=lambda x: x["score"], reverse=True)
    list_hits_dedup = []
    for hit in list_hits:
        doc_id = SegIDParser.parseSegID(hit["seg_id"]).seg_source  # 此处的doc_id对应的是SegID().seg_source
        if doc_id not in list_hits_dedup:
            # list_hits_dedup.append(doc_id)
            list_hits_dedup.append(
                {
                    "id": doc_id,
                    "score": hit["score"]
                }
            )
        if len(list_hits_dedup) > num_hits:
            break
    logger.info(f"Fetching done. [Num of original hits: {len(list_hits)}]")

    return list_hits_dedup[:num_hits]


if __name__ == "__main__":
    lecardv2_recall = {}
    searcher = load_searcher()
    searcher.num_threads = SEARCH_THREADS if SEARCH_THREADS else 1
    # with open("lecardv2_query.json", "r", encoding="utf-8") as f:
    #     for idx, line in enumerate(f):
    #         logger.info(f"Recalling for query {idx + 1}.")
    #         line = json.loads(line)
    #         q_id = line["id"]
    #         q_text = line["query"]
    #
    #         hits = BM25_Recall(q_text, RECALL_NUM_HITS, searcher)
    #         lecardv2_recall[str(q_id)] = [str(DSParser.parseDocID(hit).doc_id) for hit in hits]
    #
    #         with open("lecardv2_recall.json", "w", encoding="utf-8") as x:
    #             json.dump(lecardv2_recall, x, ensure_ascii=False)

    test_query = ("安徽省宿州市埇桥区人民检察院指控："
                  "2014年9月8日晚，被告人梅某与孔某、曾某等人（均另处）在宿州市埇桥区科技广场“天之娇”歌吧唱歌。"
                  "9月9日凌晨1时许，被告人梅某出歌吧，在花园附近小便时，被害人蒋某、王某丙等人从此某，"
                  "被告人梅某认为王某丙看了他一眼，纠集孔某、曾某等人追至“老乡鸡饭店”附近的公交站台，"
                  "持砖头等物一起对王某甲、蒋某、王某丙殴打，并将蒋某、王某丙打伤。经鉴定，蒋某、王某丙的伤情为轻微伤。"
                  "针对上述指控，公诉机关当庭提交了鉴定意见、相关书证、被害人陈述、证人证言、被告人的供述与辩解等证据予以证实，"
                  "认为被告人梅某随意殴打他人，情节恶劣，其行为触犯了《中华人民共和国刑法》××之规定，"
                  "应当以××罪追究其刑事责任，建议对被告人梅某在××以内量刑。")
    hits = BM25_Recall(test_query, 100, searcher)
    with open("test_recall.json", "w", encoding="utf-8") as x:
        json.dump(hits, x, ensure_ascii=False, indent=4)
