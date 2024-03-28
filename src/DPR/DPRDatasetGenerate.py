import jieba

from glob import glob

from whoosh.fields import *
from whoosh.qparser import QueryParser
from whoosh.analysis import Tokenizer, Token
from whoosh.filedb.filestore import RamStorage


