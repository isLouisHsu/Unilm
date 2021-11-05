python3 -u run_decode.py \
    --model_type unilm \
    --model_name_or_path unilm_base/ \
    --model_recover_path output/model.3.bin \
    --input_file data/test_data.json \
    --output_file data/test_predict.json \
    --max_seq_length 512 \
    --do_lower_case \
    --batch_size 32 \
    --beam_size 5 \
    --max_tgt_length 128
    