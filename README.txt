Modify using the code provided by huggingface
https://github.com/huggingface/transformers


step1: Install requirements package

pip3 install -r requirements.txt

-------------------------------------------------------------------------------------------------------------
step2: Execute the following command

BERT-SQG:

paragraph level model token parameter setting : 
max_seq_length 512, doc_stride 450, max_query_length 42, max_answer_length 16

sentence level model token parameter setting :
max_seq_length 192, doc_stride 128, max_query_length 42, max_answer_length 16


===== Training the SQG model ===== 

python3 SQG_train.py \
  --bert_model ../bert-base-uncased \
  --do_train \
  --train_file data/paragraph_81K_training_data.json \
  --output_dir SQG_model_paragraph_81K/ \
  --num_train_epochs 5 \
  --train_batch_size 28 \
  --max_seq_length 512 \
  --doc_stride 450 \
  --max_answer_length 16 \
  --max_query_length 42


===== Example prediction using SQG model ===== 

python3 SQG_gen_example.py \
  --bert_model ../bert-base-uncased \
  --SQG_model SQG_model_paragraph_81K/pytorch_model.bin \
  --max_seq_length 512 \
  --doc_stride 450 \
  --max_query_length 42 \
  --max_answer_length 16

example data:
    Context: The city has a proud history of theatre. Stephen Kemble of the famous Kemble family successfully managed the original Theatre Royal, Newcastle for fifteen years (1791–1806). He brought members of his famous acting family such as Sarah Siddons and John Kemble out of London to Newcastle. Stephen Kemble guided the theatre through many celebrated seasons. The original Theatre Royal in Newcastle was opened on 21 January 1788 and was located on Mosley Street. It was demolished to make way for Grey Street, where its replacement was built.
    Answers: 1788
    Human question: When did the theater in newcastle originally open?

    Context: Harvard is a large, highly residential research university. The nominal cost of attendance is high, but the University's large endowment allows it to offer generous financial aid packages. It operates several arts, cultural, and scientific museums, alongside the Harvard Library, which is the world's largest academic and private library system, comprising 79 individual libraries with over 18 million volumes.
    Answers: Harvard Library
    Human question : What is the worlds largest academic and private library system?


==== Question generation for testing data ===== 

python3 SQG_gen_eval.py \
  --bert_model ../bert-base-uncased \
  --SQG_model SQG_model_paragraph_81K/pytorch_model.bin \
  --output_dir data \
  --predict_file data/paragraph_81K_testing_data.json \
  --max_seq_length 512 \
  --doc_stride 450 \
  --max_query_length 42 \
  --max_answer_length 16


==== Evaluation of testing data ====
install the evaluation package according to this github : https://github.com/Maluuba/nlg-eval

python3 eval.py \
  --bert_model ../bert-base-uncased \
  --eval_file data/SQG_eval.json





BERT-HLSQG:

paragraph level model token parameter setting : 
max_seq_length 512, doc_stride 450, max_query_length 42, max_answer_length 16

sentence level model token parameter setting :
max_seq_length 192, doc_stride 128, max_query_length 42, max_answer_length 16

## If training chinese model, modify the "read_data" function in each code. ##

===== Training the HLSQG model ===== 

python3 HLSQG_train.py \
  --bert_model ../bert-base-uncased \
  --do_train \
  --train_file data/paragraph_81K_training_data.json \
  --output_dir HLSQG_model_paragraph_81K/ \
  --num_train_epochs 5 \
  --train_batch_size 28 \
  --max_seq_length 512 \
  --doc_stride 465 \
  --max_query_length 42


===== Example prediction using SQG model ===== 

python3 HLSQG_gen_example.py \
  --bert_model ../bert-base-uncased \
  --SQG_model HLSQG_model_paragraph_81K/pytorch_model.bin \
  --max_seq_length 512 \
  --doc_stride 465 \
  --max_query_length 42

example data:
    Context: The city has a proud history of theatre. Stephen Kemble of the famous Kemble family successfully managed the original Theatre Royal, Newcastle for fifteen years (1791–1806). He brought members of his famous acting family such as Sarah Siddons and John Kemble out of London to Newcastle. Stephen Kemble guided the theatre through many celebrated seasons. The original Theatre Royal in Newcastle was opened on 21 January 1788 and was located on Mosley Street. It was demolished to make way for Grey Street, where its replacement was built.
    Answers: 1788
    Human question: When did the theater in newcastle originally open?

    Context: Harvard is a large, highly residential research university. The nominal cost of attendance is high, but the University's large endowment allows it to offer generous financial aid packages. It operates several arts, cultural, and scientific museums, alongside the Harvard Library, which is the world's largest academic and private library system, comprising 79 individual libraries with over 18 million volumes.
    Answers: Harvard Library
    Human question : What is the worlds largest academic and private library system?


==== Question generation for testing data ===== 

python3 HLSQG_gen_eval.py \
  --bert_model ../bert-base-uncased \
  --SQG_model HLSQG_model_paragraph_81K/pytorch_model.bin \
  --output_dir data \
  --predict_file data/paragraph_81K_testing_data.json \
  --max_seq_length 512 \
  --doc_stride 465 \
  --max_query_length 42


==== Evaluation of testing data ====
install the evaluation package according to this github : https://github.com/Maluuba/nlg-eval

python3 eval.py \
  --bert_model ../bert-base-uncased \
  --eval_file data/HLSQG_eval.json







BERT-QG:

paragraph level model token parameter setting : 
max_seq_length 512, doc_stride 450, max_query_length 42, max_answer_length 16

sentence level model token parameter setting :
max_seq_length 192, doc_stride 128, max_query_length 42, max_answer_length 16


===== Training the QG model ===== 

python3 QG_train.py \
  --bert_model ../bert-base-uncased \
  --do_train \
  --train_file data/paragraph_81K_training_data.json \
  --output_dir QG_model_paragraph_81K/ \
  --num_train_epochs 5 \
  --train_batch_size 28 \
  --max_seq_length 512 \
  --doc_stride 450 \
  --max_answer_length 16 \
  --max_query_length 42


===== Example prediction using QG model ===== 

python3 QG_gen_example.py \
  --bert_model ../bert-base-uncased \
  --SQG_model QG_model_paragraph_81K/pytorch_model.bin \
  --max_seq_length 512 \
  --doc_stride 450 \
  --max_query_length 42 \
  --max_answer_length 16

example data:
    Context: The city has a proud history of theatre. Stephen Kemble of the famous Kemble family successfully managed the original Theatre Royal, Newcastle for fifteen years (1791–1806). He brought members of his famous acting family such as Sarah Siddons and John Kemble out of London to Newcastle. Stephen Kemble guided the theatre through many celebrated seasons. The original Theatre Royal in Newcastle was opened on 21 January 1788 and was located on Mosley Street. It was demolished to make way for Grey Street, where its replacement was built.
    Answers: 1788
    Human question: When did the theater in newcastle originally open?

    Context: Harvard is a large, highly residential research university. The nominal cost of attendance is high, but the University's large endowment allows it to offer generous financial aid packages. It operates several arts, cultural, and scientific museums, alongside the Harvard Library, which is the world's largest academic and private library system, comprising 79 individual libraries with over 18 million volumes.
    Answers: Harvard Library
    Human question : What is the worlds largest academic and private library system?

==== Question generation for testing data ===== 

python3 QG_gen_eval.py \
  --bert_model ../bert-base-uncased \
  --QG_model QG_model_paragraph_81K/pytorch_model.bin \
  --output_dir data \
  --predict_file data/paragraph_81K_testing_data.json \
  --max_seq_length 512 \
  --doc_stride 450 \
  --max_query_length 42 \
  --max_answer_length 16

==== Evaluation of testing data ====
install the evaluation package according to this github : https://github.com/Maluuba/nlg-eval

python3 eval.py \
  --bert_model ../bert-base-uncased \
  --eval_file data/QG_eval.json   