import os, sys, json, logging
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


def load_model():
    logger.info(f"Loading query_encoder from {DPR_MODEL_DIR}.")
    query_encoder = BertModel.from_pretrained(os.path.join(DPR_MODEL_DIR, "query_encoder"))
    tokenizer = BertTokenizer.from_pretrained(os.path.join(DPR_MODEL_DIR, "tokenizer"))
    query_encoder.eval()
    query_encoder.to(EMBEDDING_DEVICE)
    logger.info(f"Query encoder loaded.")
    return query_encoder, tokenizer


def load_embeddings(npy_list, embs_info):
    # npy_list 是一个文件路径的列表, 每个文件 (.npy) 是一个 doc 所有 segs 的 embedding 的信息
    # embs_info 是一个字典, key 是 doc id, value 是该 doc 的所有 segs 的 id, 和 npy_list 中的索引对应
    # 返回一个可迭代的对象, 每次迭代返回 npy_list 中的一个 doc 的所有 segs 的 id (来自 embs_info) 和 embedding (来自 npy_list), 即一个 npy 文件中的所有 segs 的 id 和 embedding
    for npy_path in npy_list:
        seg_embs = np.load(npy_path)
        doc_id = os.path.basename(npy_path).replace(".npy", ".json")
        seg_ids = embs_info[doc_id]
        yield from zip(seg_ids, seg_embs)


def DPR_Search(query, list_hits, num_hits, query_encoder, tokenizer):
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
        f.close()

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
        doc_id = hit["seg_id"].split("_segment_")[0]
        if doc_id not in final_results:
            final_results.append(doc_id)
        if len(final_results) > num_hits:
            break
    logger.info(f"Fetching done. [Num of original hits: {len(final_results)}]")

    return final_results[:num_hits]


if __name__ == "__main__":
    pass
