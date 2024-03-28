import re


class DocID:
    def __init__(self, full="", doc_source="", doc_id=""):
        self.full = full  # eg. lecardv2_114
        self.doc_source = doc_source  # eg. lecardv2
        self.doc_id = doc_id  # eg. 114


class SegID:
    def __init__(self, full="", doc_source="", seg_source="", doc_id="", seg_id=""):
        self.full = full  # eg. lecardv2_114_segment_514
        self.doc_source = doc_source  # eg. lecardv2
        self.seg_source = seg_source  # eg. lecardv2_114
        self.doc_id = doc_id  # eg. 114
        self.seg_id = seg_id  # eg. 514


class DSParser:  # Doc Seg ID Parser
    def __init__(self):
        pass

    @classmethod
    def parseDocID(cls, full):
        match = re.match(r"^(.+?)_(\d+)$", full)
        if match:
            doc_source, doc_id = match.groups()
            return DocID(full, doc_source, doc_id)
        return DocID(full)

    @classmethod
    def parseSegID(cls, full):
        match = re.match(r"^(.+?)_(\d+)_segment_(\d+)$", full)
        if match:
            doc_source, doc_id, seg_id = match.groups()
            seg_source = f"{doc_source}_{doc_id}"
            return SegID(full, doc_source, seg_source, doc_id, seg_id)
        return SegID(full)
