python ../../word2vec.py \
    --ds output/t2vec/train_locseq_ds.parquet \
    --output output/t2vec/word2vec.model \
    --vector_size 256 \
    --window 5 \
    --min_count 1 \
    --sg 1 \
    --workers -1 \
    --seed 42 \
    --epochs 100