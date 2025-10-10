# Note

## Development

```bash
conda create -n SR python=3.12
conda activate SR
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install jsonlines transformers sentence_transformers nltk SentencePiece protobuf accerate

git clone git@github.com:thu-coai/CTRLEval.git
git clone git@github.com:jinlanfu/GPTScore.git
```

```bash
CUDA_VISIBLE_DEVICES=0 python3 loop.py \
--input-file dataset/pubmedqa/test_data.jsonl \
--sources 'pubmedqa' \
--out-dir output \
--max-loop 3 \
--max-knowledge-loop 3 \
--max-response-loop 3 \
--demo-num 0 \
--threshold-entailment 0.8 \
--threshold-fact -1.0 \
--threshold-consistency -5
```

## References

- <https://github.com/thu-coai/CTRLEval/tree/main>
- <https://lightning.ai/docs/torchmetrics/stable/text/bert_score.html>
- <https://github.com/jinlanfu/GPTScore/blob/main/opt_score.py>
- <https://github.com/tloen/alpaca-lora/blob/main/utils/prompter.py>
