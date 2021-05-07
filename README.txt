Modify using the code provided by huggingface
https://github.com/huggingface/transformers

step0: Download MS MARCO training data from https://microsoft.github.io/msmarco/ Question Answering and Natural Langauge Generation: RETIRED(12/01/2016-10/30/2020). Prepare train and test splits using extract_selected.py

Download pytorch_model.bin for bert-base-uncased (pretrained BERT weights) from https://huggingface.co/bert-base-uncased

step1: Install requirements package

pip3 install -r requirements.txt

-------------------------------------------------------------------------------------------------------------
step2: Execute the following command

BERT-SQG:

token parameter setting : 
max_seq_length 512, doc_stride 450, max_query_length 512, max_answer_length 512


===== Training the SQG model ===== 

python3 SQG_train.py \
  --bert_model ../bert-base-uncased \
  --do_train \
  --train_file ../MARCO_train_10K.json \
  --output_dir SQG_model_MARCO_10K/ \
  --num_train_epochs 5 \
  --train_batch_size 28 \
  --max_seq_length 512 \
  --doc_stride 450 \
  --max_answer_length 512 \
  --max_query_length 512


===== Example prediction using SQG model ===== 

python3 SQG_gen_example.py \
  --bert_model ../bert-base-uncased \
  --SQG_model SQG_model_MARCO_10K/pytorch_model.bin \
  --max_seq_length 512 \
  --doc_stride 450 \
  --max_query_length 512 \
  --max_answer_length 512

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
  --SQG_model SQG_model_MARCO_10K/pytorch_model.bin \
  --output_dir . \
  --predict_file ../MARCO_test_10K.json \
  --max_seq_length 512 \
  --doc_stride 450 \
  --max_query_length 512 \
  --max_answer_length 512


==== Evaluation of testing data ====
install the evaluation package according to this github : https://github.com/Maluuba/nlg-eval

python3 eval.py \
  --bert_model ../bert-base-uncased \
  --eval_file data/SQG_eval.json

