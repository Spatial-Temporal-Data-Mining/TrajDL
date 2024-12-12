python ../../word2vec.py \
    --vector_size 250 \
    --window 5 \
    --min_count 0 \
    --sample 0.001 \
    --negative 5 \
    --alpha 0.025 \
    --hs 0 \
    --sg 1 \
    --epochs 100 \
    --workers -1 \
    --seed 42 \
    "$@"