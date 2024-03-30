import logging
import os
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

import warnings

warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import json, jsonlines

from glob import glob
from tqdm import tqdm
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import torch
import transformers
from torch.utils.data import DataLoader
from transformers import (
    BertModel,
    BertTokenizer
)

transformers.logging.set_verbosity_error()

import yaml

args = yaml.safe_load(open("config/config.yaml", "r", encoding="utf-8"))
DOCUMENTS_DIR = args["DOCUMENTS_DIR"]
DPR_MODEL_DIR = args["DPR_MODEL_DIR"]
DPR_SPLIT = args["DPR_SPLIT"]
DPR_EMBED = args["DPR_EMBED"]
DPR_SEGS_DIR = args["DPR_SEGS_DIR"]
DPR_SEGS_INFO = args["DPR_SEGS_INFO"]
DPR_SEGS_EMBS_DIR = args["DPR_SEGS_EMBS_DIR"]
EMBEDDING_BATCHSIZE = args["EMBEDDING_BATCHSIZE"]
EMBEDDING_DEVICE = args["EMBEDDING_DEVICE"]
MAX_LENGTH = args["MAX_LENGTH"]
MIN_OVERLAP = args["MIN_OVERLAP"]

from src.TextPreprocessing import DPR_TextPreProcessor


def split_database():
    """
    将原始文本库作分块，供doc-encoder进行编码（只会对未分块过的原始文本操作）
    """
    dpr_text_preprocessor = DPR_TextPreProcessor()

    if os.path.exists(DPR_SEGS_INFO):
        with open(DPR_SEGS_INFO, "r", encoding="utf-8") as f:
            dpr_segs_info = json.load(f)
    else:
        dpr_segs_info = {}

    all_docs_path = glob(os.path.join(DOCUMENTS_DIR, "*.json"))
    all_docs_path = list(
        filter(lambda path: os.path.basename(path) not in dpr_segs_info, all_docs_path)
    )

    logger.info(f"Splitting new documents into segments. [Num of new docs: {len(all_docs_path)}]")
    for doc_path in tqdm(all_docs_path):
        doc_basename = os.path.basename(doc_path)
        doc_segs = dpr_text_preprocessor.docfile2segs(doc_path, MAX_LENGTH, MIN_OVERLAP)
        with jsonlines.open(os.path.join(DPR_SEGS_DIR, doc_basename), "w") as writer:
            for doc_seg in doc_segs:
                writer.write(doc_seg)
        dpr_segs_info[doc_basename] = len(doc_segs)
        # TODO?
        with open(DPR_SEGS_INFO, "w", encoding="utf-8") as f:
            json.dump(dpr_segs_info, f, ensure_ascii=False, indent=4)
    logger.info(
        f"Splitting done. [Num of all docs: {len(dpr_segs_info)}; Num of all segs: {sum(dpr_segs_info.values())}]")


def embed_database():
    """
    对已经分块的文本进行编码，不同checkpoint的编码结果分开保存（只会对当前checkpoint未编码过的文本操作）
    """
    logger.info(f"Loading doc_encoder from {DPR_MODEL_DIR}.")
    doc_encoder = BertModel.from_pretrained(os.path.join(DPR_MODEL_DIR, "doc_encoder"))
    tokenizer = BertTokenizer.from_pretrained(os.path.join(DPR_MODEL_DIR, "doc_encoder"))
    doc_encoder.eval()
    doc_encoder.to(EMBEDDING_DEVICE)
    logger.info(f"Doc_encoder loaded.")

    ckpt_name = os.path.basename(DPR_MODEL_DIR)
    os.makedirs(os.path.join(DPR_SEGS_EMBS_DIR, ckpt_name), exist_ok=True)

    info_path = os.path.join(DPR_SEGS_EMBS_DIR, f"{ckpt_name}_info.json")
    if os.path.exists(info_path):
        with open(info_path, "r", encoding="utf-8") as f:
            dpr_embeddings_info = json.load(f)
    else:
        dpr_embeddings_info = {}

    all_docs_path = [
        path for path in glob(os.path.join(DPR_SEGS_DIR, "*.json"))
        if os.path.basename(path) not in dpr_embeddings_info
    ]

    logger.info(f"Embedding new segments. [Num of new docs: {len(all_docs_path)}]")
    for doc_path in tqdm(all_docs_path):
        doc_basename = os.path.basename(doc_path)

        doc_ids = []
        doc_embeddings = []

        with jsonlines.open(doc_path, "r") as reader:
            doc_segs = list(reader)

        for batch in DataLoader(doc_segs, batch_size=EMBEDDING_BATCHSIZE, shuffle=False, collate_fn=lambda x: x):
            doc_ids.extend([seg["id"] for seg in batch])
            text_batch = [seg["contents"] for seg in batch]

            model_inputs = tokenizer(
                text_batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH
            ).to(EMBEDDING_DEVICE)

            with torch.no_grad():
                model_outputs = doc_encoder(**model_inputs)

            embedding_batch = model_outputs.last_hidden_state[:, 0, :].cpu().numpy()
            doc_embeddings.extend(embedding_batch)

        doc_embeddings = np.array(doc_embeddings)

        npy_doc_basename = doc_basename.replace(".json", ".npy")
        np.save(os.path.join(DPR_SEGS_EMBS_DIR, ckpt_name, npy_doc_basename), doc_embeddings)
        dpr_embeddings_info[doc_basename] = doc_ids

        with open(info_path, "w", encoding="utf-8") as f:
            json.dump(dpr_embeddings_info, f, ensure_ascii=False, indent=4)
    logger.info(
        f"Embedding done. "
        f"[Num of all docs: {len(dpr_embeddings_info)}; "
        f"Num of all segs: {sum([len(ids) for ids in dpr_embeddings_info.values()])}]"
    )


def main(split, embed):
    if split:
        split_database()
    if embed:
        embed_database()


if __name__ == "__main__":
    main(DPR_SPLIT, DPR_EMBED)
