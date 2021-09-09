python3 main_run.py \
  --filename ../ddi_dataset/bc5cdr \
  --data_dir ../ddi_dataset/bc5cdr  \
  --label_file ../ddi_dataset/bc5cdr/labels.txt  \
  --model_dir ../saved_model/BC5CDR_biobert_01 \
  --model_type bert  \
  --model only_bert  \
  --per_gpu_train_batch_size=8  \
  --per_gpu_eval_batch_size=16  \
  --max_steps=-1  \
  --num_train_epochs=10 \
  --gradient_accumulation_steps=1  \
  --learning_rate=5e-5  \
  --logging_steps=200  \
  --save_steps=100  \
  --adam_epsilon=1e-8  \
  --warmup_steps=0  \
  --dropout_rate=0.1  \
  --weight_decay=0.0  \
  --seed=42  \
  --max_grad_norm=1.0  \
  --do_test \
  --do_train \
  --do_eval \

python3 main_run.py \
  --filename ../ddi_dataset/NCBI-disease \
  --data_dir ../ddi_dataset/NCBI-disease  \
  --label_file ../ddi_dataset/NCBI-disease/labels.txt  \
  --model_dir ../saved_model/NCBI-disease_biobert_01 \
  --model_type bert  \
  --model only_bert  \
  --per_gpu_train_batch_size=8  \
  --per_gpu_eval_batch_size=16  \
  --max_steps=-1  \
  --num_train_epochs=10 \
  --gradient_accumulation_steps=1  \
  --learning_rate=5e-5  \
  --logging_steps=200  \
  --save_steps=100  \
  --adam_epsilon=1e-8  \
  --warmup_steps=0  \
  --dropout_rate=0.1  \
  --weight_decay=0.0  \
  --seed=42  \
  --max_grad_norm=1.0  \
  --do_test \
  --do_train \
  --do_eval \




