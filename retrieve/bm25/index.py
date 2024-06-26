import logging
import os
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

import warnings

warnings.filterwarnings("ignore", category=UserWarning)

import subprocess
import json, jsonlines

from glob import glob
from tqdm import tqdm
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import yaml

args = yaml.safe_load(open("config/config.yaml", "r", encoding="utf-8"))
DOCUMENTS_DIR = args["DOCUMENTS_DIR"]
STOPWORD_FILE_DIR = args["STOPWORD_FILE_DIR"]
BM25_TOKENIZE = args["BM25_TOKENIZE"]
BM25_INDEX = args["BM25_INDEX"]
BM25_TEXTS_DIR = args["BM25_TEXTS_DIR"]
BM25_TEXTS_INFO = args["BM25_TEXTS_INFO"]
BM25_TEXTS_INDEX_DIR = args["BM25_TEXTS_INDEX_DIR"]
INDEX_THREADS = args["INDEX_THREADS"]

from src.TextPreprocessing import BM25_TextPreProcessor


def tokenize_database():
    """
    将原始文本库作分词拼接处理，供pyserini进行全文索引（只会对未分块过的原始文本操作）
    """
    bm25_text_preprocessor = BM25_TextPreProcessor()
    bm25_text_preprocessor.get_stopwords(STOPWORD_FILE_DIR)

    if os.path.exists(BM25_TEXTS_INFO):
        with open(BM25_TEXTS_INFO, "r", encoding="utf-8") as f:
            bm25_texts_info = json.load(f)
    else:
        bm25_texts_info = {}

    all_docs_path = [
        path for path in glob(os.path.join(DOCUMENTS_DIR, "*.json"))
        if os.path.basename(path) not in bm25_texts_info
    ]

    logger.info(f"Tokenizing new documents. [Num of new docs: {len(all_docs_path)}]")
    for doc_path in tqdm(all_docs_path):
        doc_basename = os.path.basename(doc_path)
        tokenized_doc = bm25_text_preprocessor.tokenize_docfile(doc_path)
        with jsonlines.open(os.path.join(BM25_TEXTS_DIR, doc_basename), "w") as writer:
            writer.write(tokenized_doc)
        bm25_texts_info[doc_basename] = True
        # TODO?
        with open(BM25_TEXTS_INFO, "w", encoding="utf-8") as f:
            json.dump(bm25_texts_info, f, ensure_ascii=False, indent=4)
    logger.info(f"Tokenizing done. [Num of all docs: {len(bm25_texts_info)}")


def index_database():
    """
    清除原先存在的索引文件，对已经分词拼接的文本重新索引
    """
    logger.info(f"Cleaning previous index files.")
    for file in glob(os.path.join(BM25_TEXTS_INDEX_DIR, "*")):
        os.remove(file)
    logger.info(f"Previous index files cleaned.")

    logger.info(f"Creating new pyserini index files.")
    cmd = [
        "python", "-m", "pyserini.index.lucene",
        "-collection", "JsonCollection",
        "-generator", "DefaultLuceneDocumentGenerator",
        "-threads", str(INDEX_THREADS),
        "-input", BM25_TEXTS_DIR,
        "-index", BM25_TEXTS_INDEX_DIR,
        "-storePositions",
        "-storeDocvectors",
        "-storeRaw"
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    logger.info(result.stdout.decode("utf-8"))
    if result.stderr:
        logger.error(result.stderr.decode("utf-8"))
    logger.info(f"New pyserini index files created.")


def main(tokenize, index):
    if tokenize:
        tokenize_database()
    if index:
        index_database()


if __name__ == "__main__":
    main(BM25_TOKENIZE, BM25_INDEX)
