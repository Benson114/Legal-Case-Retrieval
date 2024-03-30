import json
import logging
import os
import sys

import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

from pathlib import Path

import faiss
import torch
import transformers

from transformers import (
    BertModel,
    BertTokenizer
)

import yaml

args = yaml.safe_load(open("config/config.yaml", "r", encoding="utf-8"))
DPR_MODEL_DIR = args["DPR_MODEL_DIR"]
DPR_SEGS_DIR = args["DPR_SEGS_DIR"]
DPR_SEGS_INFO = args["DPR_SEGS_INFO"]
DPR_SEGS_EMBS_DIR = args["DPR_SEGS_EMBS_DIR"]
EMBEDDING_DEVICE = args["EMBEDDING_DEVICE"]
MAX_LENGTH = args["MAX_LENGTH"]

transformers.logging.set_verbosity_error()

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

from src.TextPreprocessing import DPR_TextPreProcessor
from src.DSParser import DSParser


def load_model():
    logger.info(f"Loading query_encoder from {DPR_MODEL_DIR}.")
    query_encoder = BertModel.from_pretrained(os.path.join(DPR_MODEL_DIR, "query_encoder"))
    tokenizer = BertTokenizer.from_pretrained(os.path.join(DPR_MODEL_DIR, "query_encoder"))
    query_encoder.eval()
    query_encoder.to(EMBEDDING_DEVICE)
    logger.info(f"Query encoder loaded.")
    return query_encoder, tokenizer


def load_embeddings(npy_list, embs_info):
    # npy_list是一个文件路径的列表，每个文件（.npy）是一个doc所有segs的embedding的信息
    # embs_info是一个字典，key是doc_id，value是该doc的所有segs的id，和npy_list中的索引对应
    # 返回一个可迭代的对象，每次迭代返回npy_list中的一个doc所有segs的id（来自embs_info）和embedding（来自npy_list）
    # 即一个npy文件中的所有segs的id和embedding
    for npy_path in npy_list:
        seg_embs = np.load(npy_path)
        doc_id = os.path.basename(npy_path).replace(".npy", ".json")
        seg_ids = embs_info[doc_id]
        yield seg_ids, seg_embs


def DPR_Search(query, list_hits, num_hits, query_encoder, tokenizer):
    """
    :param query: 查询文本/文档，是一个字符串
    :param list_hits: 初步召回结果/候选池列表，每个元素是一个字典，包含id和score两个key
    :param num_hits: 最终检索结果的数量
    :param query_encoder: 加载的query_encoder模型
    :param tokenizer: 加载的tokenizer
    :return: 最终检索结果的列表，每个元素是一个字典，包含id和score两个key
    """
    dpr_text_preprocessor = DPR_TextPreProcessor()

    logger.info(f"Splitting query into segments.")
    query_segs = dpr_text_preprocessor.doc2segs(query, MAX_LENGTH, 0)
    logger.info(f"Splitting done. [Num of query segs: {len(query_segs)}]")

    ckpt_name = os.path.basename(DPR_MODEL_DIR)
    embs_dir = os.path.join(DPR_SEGS_EMBS_DIR, ckpt_name)
    embs_info_dir = os.path.join(DPR_SEGS_EMBS_DIR, f"{ckpt_name}_info.json")
    npy_list = [
        os.path.join(embs_dir, f"{hit['id']}.npy")
        for hit in list_hits
    ]  # Recall results
    with open(embs_info_dir, "r", encoding="utf-8") as f:
        embs_info = json.load(f)  # Embs info of segs of each doc

    logger.info(f"Embedding query segments.")
    query_embs = []
    for seg in query_segs:
        tokenized_seg = tokenizer(seg, return_tensors="pt", padding="max_length", truncation=True,
                                  max_length=MAX_LENGTH).to(EMBEDDING_DEVICE)
        with torch.no_grad():
            seg_emb = query_encoder(**tokenized_seg).last_hidden_state[:, 0, :].cpu().numpy()
        query_embs.append(seg_emb)
    query_embs = np.concatenate(query_embs, axis=0)
    logger.info(f"Embedding done. [Num of query segs: {len(query_embs)}]")

    logger.info("Preliminary searching for relevant documents.")
    search_results = []
    for seg_ids, seg_embs in load_embeddings(npy_list, embs_info):
        faiss_index = faiss.IndexFlatIP(seg_embs.shape[1])
        faiss_index.add(seg_embs)
        D, I = faiss_index.search(query_embs, 1)
        for _, (d, i) in enumerate(zip(D, I)):
            search_results.append(
                {
                    "seg_id": seg_ids[i[0]],
                    "score": d[0]
                }
            )
    search_results.sort(key=lambda x: x["score"], reverse=True)
    logger.info(f"Preliminary searching done. [Num of segments: {len(search_results)}]")

    logger.info("Fetching original searching results.")
    final_results = []
    for hit in search_results:
        doc_id = DSParser.parseSegID(hit["seg_id"]).seg_source  # 此处的doc_id对应的是SegID().seg_source
        if doc_id not in final_results:
            # final_results.append(doc_id)
            final_results.append(
                {
                    "id": doc_id,
                    "score": float(hit["score"])  # 将numpy.float32转为float，避免写入json时报错
                }
            )
        if len(final_results) > num_hits:
            break
    logger.info(f"Fetching done. [Num of original hits: {len(final_results)}]")

    return final_results[:num_hits]


if __name__ == "__main__":
    test_query = ("安徽省宿州市埇桥区人民检察院指控："
                  "2014年9月8日晚，被告人梅某与孔某、曾某等人（均另处）在宿州市埇桥区科技广场“天之娇”歌吧唱歌。"
                  "9月9日凌晨1时许，被告人梅某出歌吧，在花园附近小便时，被害人蒋某、王某丙等人从此某，"
                  "被告人梅某认为王某丙看了他一眼，纠集孔某、曾某等人追至“老乡鸡饭店”附近的公交站台，"
                  "持砖头等物一起对王某甲、蒋某、王某丙殴打，并将蒋某、王某丙打伤。经鉴定，蒋某、王某丙的伤情为轻微伤。"
                  "针对上述指控，公诉机关当庭提交了鉴定意见、相关书证、被害人陈述、证人证言、被告人的供述与辩解等证据予以证实，"
                  "认为被告人梅某随意殴打他人，情节恶劣，其行为触犯了《中华人民共和国刑法》××之规定，"
                  "应当以××罪追究其刑事责任，建议对被告人梅某在××以内量刑。")
    list_hits = json.load(open("test_recall.json", "r", encoding="utf-8"))
    num_hits = 5
    query_encoder, tokenizer = load_model()
    final_results = DPR_Search(test_query, list_hits, num_hits, query_encoder, tokenizer)
    with open("test_search.json", "w", encoding="utf-8") as f:
        json.dump(final_results, f, ensure_ascii=False, indent=4)
