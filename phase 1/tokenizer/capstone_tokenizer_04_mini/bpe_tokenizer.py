
# bpe_tokenizer.py — self-contained byte-level BPE tokenizer.
# Drop this file next to an `artifacts/` folder and you're done.

import os, re, json

PAD, UNK, BOS, EOS, SEP, CLS, MASK = 0, 1, 2, 3, 4, 5, 6
RES_START,   RES_N   = 7,  16
EXTRA_START, EXTRA_N = 23, 100
BYTE_START  = 123
MERGE_START = 379
VOCAB_SIZE  = 32000

SPECIAL_NAMES = ["<pad>", "<unk>", "<bos>", "<eos>", "<sep>", "<cls>", "<mask>"] \
              + [f"<res_{i}>" for i in range(RES_N)] \
              + [f"<extra_id_{i}>" for i in range(EXTRA_N)]
SPECIAL_IDS = list(range(len(SPECIAL_NAMES)))

PRETOK_RE = re.compile(
    r" ?[^\W\d_]+"
    r"| ?\d"
    r"| ?(?:[^\s\w]|_)+"
    r"|\s+(?!\S)"
    r"|\s+"
)


class BPETokenizer:
    def __init__(self, id_to_bytes, merges, special_names=None,
                 pretok_re=None, config=None):
        self.id_to_bytes = id_to_bytes
        self.merges = list(merges)
        self.merge_rank = {p: i for i, p in enumerate(self.merges)}
        self.pair_to_id = {p: MERGE_START + i for i, p in enumerate(self.merges)}
        self.special_names = list(special_names or SPECIAL_NAMES)
        self.special_to_id = {n: i for i, n in enumerate(self.special_names)}
        pat = "|".join(re.escape(s) for s in
                       sorted(self.special_names, key=len, reverse=True))
        self.special_re = re.compile(pat)
        self.pretok_re = pretok_re or PRETOK_RE
        self.config = config or {}
        self._cache = {}

    def _encode_pretok_bytes(self, bs):
        c = self._cache.get(bs)
        if c is not None:
            return c
        ids = [BYTE_START + b for b in bs]
        while len(ids) > 1:
            best_r, best_i = None, -1
            for i in range(len(ids) - 1):
                r = self.merge_rank.get((ids[i], ids[i+1]))
                if r is not None and (best_r is None or r < best_r):
                    best_r, best_i = r, i
            if best_i < 0:
                break
            nid = self.pair_to_id[(ids[best_i], ids[best_i+1])]
            ids = ids[:best_i] + [nid] + ids[best_i+2:]
        if len(self._cache) > 1_000_000:
            self._cache.clear()
        self._cache[bs] = ids
        return ids

    def encode(self, text, add_bos=False, add_eos=False,
               allow_special=False, return_offsets=False):
        ids, offs = [], []

        segs = []
        if allow_special:
            last = 0
            for m in self.special_re.finditer(text):
                if m.start() > last:
                    segs.append(("raw", text[last:m.start()], last))
                segs.append(("sp", m.group(0), m.start()))
                last = m.end()
            if last < len(text):
                segs.append(("raw", text[last:], last))
        else:
            segs.append(("raw", text, 0))

        for kind, seg, base in segs:
            if kind == "sp":
                ids.append(self.special_to_id[seg])
                if return_offsets:
                    offs.append((base, base + len(seg)))
                continue

            for m in self.pretok_re.finditer(seg):
                cs, ce = m.start(), m.end()
                sub = seg[cs:ce]
                sub_ids = self._encode_pretok_bytes(sub.encode("utf-8"))

                if not return_offsets:
                    ids.extend(sub_ids)
                    continue

                b2c = []
                for ci, ch in enumerate(sub):
                    b2c.extend([ci] * len(ch.encode("utf-8")))
                bp = 0
                for tid in sub_ids:
                    tb = self.id_to_bytes[tid]
                    b0, b1 = bp, bp + len(tb)
                    bp = b1
                    c0 = b2c[b0] if b0 < len(b2c) else len(sub)
                    c1 = (b2c[b1 - 1] + 1) if (b1 > 0 and b1 - 1 < len(b2c)) \
                         else len(sub)
                    ids.append(tid)
                    offs.append((base + cs + c0, base + cs + c1))

        if add_bos:
            ids = [BOS] + ids
            if return_offsets:
                offs = [(0, 0)] + offs
        if add_eos:
            ids = ids + [EOS]
            if return_offsets:
                offs = offs + [(len(text), len(text))]

        return (ids, offs) if return_offsets else ids

    def decode(self, ids, skip_special=True):
        parts = []
        for i in ids:
            if skip_special and 0 <= i < len(self.special_names):
                continue
            b = self.id_to_bytes.get(i)
            if b is None:
                continue
            parts.append(b)
        return b"".join(parts).decode("utf-8", errors="replace")

    def truncate_words(self, text, n):
        ws = list(re.finditer(r"\S+", text))
        if len(ws) <= n:
            return text
        return text[:ws[n - 1].end()]

    def seq_decoder_only(self, text):
        return [BOS] + self.encode(text) + [EOS]

    def seq_enc_dec(self, src, tgt):
        enc = self.encode(src) + [EOS]
        dec = [BOS] + self.encode(tgt)
        lab = self.encode(tgt) + [EOS]
        return enc, dec, lab

    def seq_pair(self, a, b):
        return [CLS] + self.encode(a) + [SEP] + self.encode(b) + [SEP]

    @property
    def vocab_size(self):   return VOCAB_SIZE
    @property
    def pad_id(self):       return PAD
    @property
    def unk_id(self):       return UNK
    @property
    def bos_id(self):       return BOS
    @property
    def eos_id(self):       return EOS
    @property
    def sep_id(self):       return SEP
    @property
    def cls_id(self):       return CLS
    @property
    def mask_id(self):      return MASK
    @property
    def res_ids(self):
        return list(range(RES_START, RES_START + RES_N))
    @property
    def sentinel_ids(self):
        return list(range(EXTRA_START, EXTRA_START + EXTRA_N))
    @property
    def byte_ids(self):
        return list(range(BYTE_START, BYTE_START + 256))


def save_tokenizer(tok, path, extra_cfg=None):
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "merges.txt"), "w", encoding="utf-8") as f:
        for a, b in tok.merges:
            f.write(f"{tok.id_to_bytes[a].hex()} {tok.id_to_bytes[b].hex()}\n")
    with open(os.path.join(path, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump({str(i): tok.id_to_bytes[i].hex() for i in tok.id_to_bytes}, f)
    cfg = {
        "format_version": 1,
        "vocab_size": VOCAB_SIZE,
        "num_merges": len(tok.merges),
        "specials": {n: i for i, n in enumerate(tok.special_names)},
        "pretok_regex": tok.pretok_re.pattern,
        "byte_start": BYTE_START,
        "merge_start": MERGE_START,
        "extra_start": EXTRA_START,
        "res_start": RES_START,
        "frozen": True,
    }
    if extra_cfg:
        cfg.update(extra_cfg)
    with open(os.path.join(path, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    return path


def load_tokenizer(path, max_merges=None):
    with open(os.path.join(path, "vocab.json")) as f:
        vocab_json = json.load(f)
    with open(os.path.join(path, "config.json")) as f:
        cfg = json.load(f)

    id_to_bytes = {int(k): bytes.fromhex(v) for k, v in vocab_json.items()}

    bytes_to_id = {}
    for i, b in id_to_bytes.items():
        bytes_to_id.setdefault(b, i)

    merges = []
    with open(os.path.join(path, "merges.txt")) as f:
        for line in f:
            a_hex, b_hex = line.strip().split()
            a_id = bytes_to_id[bytes.fromhex(a_hex)]
            b_id = bytes_to_id[bytes.fromhex(b_hex)]
            merges.append((a_id, b_id))

    if max_merges is not None:
        merges = merges[:max_merges]
        keep = set(SPECIAL_IDS) \
             | set(range(BYTE_START, BYTE_START + 256)) \
             | set(range(MERGE_START, MERGE_START + len(merges)))
        id_to_bytes = {i: b for i, b in id_to_bytes.items() if i in keep}

    pretok_re = PRETOK_RE
    if cfg.get("pretok_regex"):
        pretok_re = re.compile(cfg["pretok_regex"])

    return BPETokenizer(id_to_bytes, merges,
                        special_names=SPECIAL_NAMES,
                        pretok_re=pretok_re,
                        config=cfg)


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("usage: python bpe_tokenizer.py /path/to/artifacts/")
        sys.exit(1)
    t = load_tokenizer(sys.argv[1])
    s = "Section 302 of the Indian Penal Code, 1860."
    ids = t.encode(s)
    assert t.decode(ids) == s
    print(f"OK  vocab={t.vocab_size}  merges={len(t.merges)}")
