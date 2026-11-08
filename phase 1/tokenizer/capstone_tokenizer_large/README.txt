Capstone byte-level BPE tokenizer
==================================

vocab_size:  32000
num_merges:  31621
trained_at:  2026-10-02 17:26:04

Usage in a new notebook:

    # (a) Add Data -> Upload Data -> upload capstone_tokenizer.zip
    # (b) unzip it (or read files directly from /kaggle/input/)
    import importlib.util, sys
    p = '/kaggle/input/capstone-tokenizer/bpe_tokenizer.py'
    spec = importlib.util.spec_from_file_location('bpe_tokenizer', p)
    m = importlib.util.module_from_spec(spec)
    sys.modules['bpe_tokenizer'] = m
    spec.loader.exec_module(m)
    tok = m.load_tokenizer('/kaggle/input/capstone-tokenizer/artifacts')

    ids = tok.encode('Section 302 of the Indian Penal Code, 1860.')
    print(tok.decode(ids))
