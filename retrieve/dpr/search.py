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
SEARCH_NUM_HITS = args["SEARCH_NUM_HITS"]

transformers.logging.set_verbosity_error()

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

from src.TextPreprocessing import DPR_TextPreProcessor
from src.SegIDParser import SegIDParser


def load_model():
    logger.info(f"Loading query_encoder from {DPR_MODEL_DIR}.")
    query_encoder = BertModel.from_pretrained(os.path.join(DPR_MODEL_DIR, "query_encoder"))
    tokenizer = BertTokenizer.from_pretrained(os.path.join(DPR_MODEL_DIR, "query_encoder"))
    query_encoder.eval()
    query_encoder.to(EMBEDDING_DEVICE)
    logger.info(f"Query encoder loaded.")
    return query_encoder, tokenizer


def load_embeddings(npy_list, embs_info):
    """
    :param npy_list: embedding文件的路径列表，每个文件是一个doc所有segs的embedding的信息
    :param embs_info: 每个doc的所有segs的id，和npy_list中的索引对应
    :return: 生成器，每次返回一个doc的所有segs的id和embedding，即一个npy文件中所有segs的id和embedding
    """
    for npy_path in npy_list:
        seg_embs = np.load(npy_path)
        doc_id = os.path.basename(npy_path).replace(".npy", ".json")
        seg_ids = embs_info[doc_id]
        yield seg_ids, seg_embs


def DPR_Search(query, list_hits, num_hits, query_encoder, tokenizer):
    """
    :param query: 查询文本/文档，是一个字符串
    :param list_hits: 初步召回结果/候选池列表，每个元素是一个字典，至少包含id这个key
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
        # os.path.join(embs_dir, f"{hit['id']}.npy")
        os.path.join(embs_dir, f"{hit}.npy")
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
        doc_id = SegIDParser.parseSegID(hit["seg_id"]).seg_source  # 此处的doc_id对应的是SegID().seg_source
        if doc_id not in final_results:
            # final_results.append(doc_id)
            final_results.append(
                {
                    "id": doc_id,
                    "score": float(hit["score"])  # 将numpy.float32转为float，避免写入json时报错
                }
            )
        if len(final_results) >= num_hits:
            break
    logger.info(f"Fetching done. [Num of original hits: {len(final_results)}]")

    return final_results[:num_hits]


if __name__ == "__main__":
    query_path = "eval/cail2022_test_recall.json"
    cail2022_res = {}
    query_encoder, tokenizer = load_model()
    with open(query_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            logger.info(f"Searching for query {idx + 1}.")
            line = json.loads(line)
            q_id = line["id"]
            q_text = line["text"]
            q_hits = line["recall"]

            list_hits = []
            search_results = DPR_Search(q_text, q_hits, SEARCH_NUM_HITS, query_encoder, tokenizer)

            search_results = [res["id"].replace("cail2022_", "") for res in search_results]
            cail2022_res[q_id] = search_results
    with open("eval/prediction.json", "w", encoding="utf-8") as f:
        json.dump(cail2022_res, f, ensure_ascii=False)
