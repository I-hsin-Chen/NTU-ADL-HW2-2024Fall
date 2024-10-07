accelerate launch inference.py \
--model_name_or_path "model" \
--source_prefix "summarize: " \
--text_column maintext \
--gradient_accumulation_steps 2 \
--max_source_length 512 \
--max_target_length 128 \
--testing_file $1 \
--output_path $2 \
--num_beams 1
