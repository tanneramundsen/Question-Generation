import torch
import torch.nn.functional as F
from HL_tokenization import whitespace_tokenize, BasicTokenizer, BertTokenizer
from modeling import BertForGenerativeSeq
import collections
import logging
import os
import json
from tqdm import tqdm, trange
import argparse

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

class Data(object):
    """A single training/test example for the Squad dataset."""

    def __init__(self,
                 doc_tokens,
                 answers_text,
                 answer_start):
        self.doc_tokens = doc_tokens
        self.answers_text = answers_text
        self.answer_start = answer_start

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 tokens,
                 input_ids,
                 input_mask,
                 segment_ids,
                 label_pos):
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_pos = label_pos

def read_data(context, answer, answer_start):

    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    datas = []
    answer_text = answer
    answer_start = answer_start

    answer_len = len(answer_text) 


    doc_tokens = []
    prev_is_whitespace = True
    for i, c in enumerate(context):
        if is_whitespace(c):
            prev_is_whitespace = True
        else:
# -----------------------------------------------------------------------------------------
# If training chinese models, please use this area code and comments the english area
                #chinese 
#                 if len(answer_text) == 1 and i == answer_start:
#                     c = "[HL]" + c + "[HL]"
#                 elif answer_text[0] == c and i == answer_start:
#                     c = "[HL]" + c
#                 elif answer_text[-1] == c and i == answer_start + answer_len - 1:
#                     c = c + "[HL]"
# -----------------------------------------------------------------------------------------

            #eng
            if len(answer_text) == 1 and i == answer_start:
                doc_tokens.append("[HL]")
                doc_tokens.append(c)
                doc_tokens.append("[HL]")
                prev_is_whitespace = True
                continue
            elif answer_text[0] == c and i == answer_start:
                doc_tokens.append("[HL]")
                doc_tokens.append(c)
                prev_is_whitespace = False
                continue
            elif answer_text[-1] == c and i == answer_start + answer_len - 1:
                if prev_is_whitespace:
                    doc_tokens.append(c)
                else:
                    doc_tokens[-1] += c
                                
                doc_tokens.append("[HL]")
                prev_is_whitespace = False
                continue
# -----------------------------------------------------------------------------------------

            if prev_is_whitespace:
                doc_tokens.append(c)
            else:
                doc_tokens[-1] += c
            prev_is_whitespace = False

    
    data = Data(
        doc_tokens=doc_tokens,
        answers_text=answer_text,
        answer_start=answer_start)
    datas.append(data)
        
    return datas


def convert_data_to_features(data, tokenizer, max_seq_length,
                                 doc_stride, max_query_length):
    
    features = []

    answer_tokens = tokenizer.tokenize(data[0].answers_text)

    all_doc_tokens = []                 
    tok_to_orig_index = []              
    orig_to_tok_index = []              
    
    for (i, token) in enumerate(data[0].doc_tokens):
        orig_to_tok_index.append(len(all_doc_tokens))
        sub_tokens = tokenizer.tokenize(token)
        for sub_token in sub_tokens:
            tok_to_orig_index.append(i)
            all_doc_tokens.append(sub_token)


    # The -5 accounts for [CLS], [HL], [HL], [SEP] and [SEP]
    max_tokens_for_doc = max_seq_length - max_query_length - 5 

    # We can have documents that are longer than the maximum sequence length.
    # To deal with this we do a sliding window approach, where we take chunks
    # of the up to our max length with a stride of `doc_stride`.
    _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
        "DocSpan", ["start", "length"])
    doc_spans = []
    start_offset = 0
    while start_offset < len(all_doc_tokens):
        length = len(all_doc_tokens) - start_offset
        if length > max_tokens_for_doc:
            length = max_tokens_for_doc
        doc_spans.append(_DocSpan(start=start_offset, length=length))
        if start_offset + length == len(all_doc_tokens):
            break
        start_offset += min(length, doc_stride)
    
    for (doc_span_index, doc_span) in enumerate(doc_spans):
        tokens = []
        token_to_orig_map = {}
        token_is_max_context = {}
        segment_ids = []


        tokens.append("[CLS]")
        segment_ids.append(0)
        
        for i in range(doc_span.length):
            split_token_index = doc_span.start + i
            token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]

            is_max_context = _check_is_max_context(doc_spans, doc_span_index,
                                                   split_token_index)
            token_is_max_context[len(tokens)] = is_max_context
            tokens.append(all_doc_tokens[split_token_index])
            segment_ids.append(0)
        
        tokens.append("[SEP]")
        segment_ids.append(0)

        label_pos = len(tokens)         

        tokens.append("[MASK]")
        segment_ids.append(2)            


        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # check [HL]
        check_symbol = 0
        for token_index, token_id in enumerate(input_ids):
            if token_id == 99:
                check_symbol += 1
                segment_ids[token_index] = 1
                continue

            elif check_symbol == 1:
                segment_ids[token_index] = 1
        
        if len(doc_spans) > 1 and check_symbol != 2:
            print("symbol error")
            if doc_span_index == len(doc_spans) - 1:
                if check_symbol == 0:
                    print(data[0].doc_tokens)
                    print(data[0].answers_text)
                    print(check_symbol)
                    print(input_ids)
                    print('HL error')
                    exit()
                    
                else:     
                    insert_num = max_tokens_for_doc - len(input_ids) + 3 #[CLS] [SEP] [MASK]

                    for num, pre_input_id in enumerate(pre_input_ids[-insert_num - 2:-2]):
                        input_ids.insert(num + 1,pre_input_id)

                    segment_ids = []
                    segment_ids = [0] * len(input_ids)

                    check_symbol = 0
                    for token_index, token_id in enumerate(input_ids):
                        if token_id == 99:
                            check_symbol += 1
                            segment_ids[token_index] = 1
                            continue

                        elif check_symbol == 1:
                            segment_ids[token_index] = 1

                    segment_ids[-1] = 2
            else:
                pre_input_ids = deepcopy(input_ids)
                continue

        input_mask = [1] * len(input_ids)
        
        
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)


        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        features.append(
            InputFeatures(
                tokens=tokens,
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                label_pos = label_pos
                ))
        break

    return features

def _check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""

    # Because of the sliding window approach taken to scoring documents, a single
    # token can appear in multiple documents. E.g.
    #  Doc: the man went to the store and bought a gallon of milk
    #  Span A: the man went to the
    #  Span B: to the store and bought
    #  Span C: and bought a gallon of
    #  ...
    #
    # Now the word 'bought' will have two scores from spans B and C. We only
    # want to consider the score with "maximum context", which we define as
    # the *minimum* of its left and right context (the *sum* of left and
    # right context will always be the same, of course).
    #
    # In the example the maximum context for 'bought' would be span C since
    # it has 1 left context and 3 right context, while span B has 4 left context
    # and 0 right context.
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index

def BS(model, tokenizer, input_ids, segment_ids, input_mask, input_num, beam_search_num, per_data=None, min_query_length = 3):

    res = []
    predictions = model(input_ids.unsqueeze(0), segment_ids.unsqueeze(0), input_mask.unsqueeze(0))
    
    predictions_SM = F.log_softmax(predictions[0][input_num],0) 
    
    while(len(res) < beam_search_num):
        
        predicted_index = torch.argmax(predictions_SM).item()
        sorce = predictions_SM[predicted_index].item()
        
        predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])
        predicted_text = predicted_token[0]
        order = str(len(res))
        
        predictions_SM[predicted_index] = -1000000000

        if (per_data == None or per_data[4] < min_query_length) and predicted_text == '[SEP]':   ## limit the minimum length of the question
            # print('min error')                                                                 ## if true , select the next word
            continue

        if per_data != None  and predicted_index in per_data[-1:]:   ##limit repeat word generation
            # print('repeat error')
            continue

        if per_data == None:
            res.append((sorce, [predicted_index], predicted_text, order, 1))
        else:
            
            predicted_index_list = per_data[1] + [predicted_index]
            
            sorce = per_data[0] + sorce 
            
            if '##' in predicted_text:
                predicted_text = per_data[2] + predicted_text.replace('##','')
            else:
                predicted_text = per_data[2] + ' ' + predicted_text

            order = per_data[3] + order
            step = per_data[4] + 1

            res.append((sorce, predicted_index_list, predicted_text, order, step))

    return res



def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")

    parser.add_argument("--SQG_model", default=None, type=str, required=True,
                        help="SQG_model path")

    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model checkpoints and predictions will be written.")

    parser.add_argument("--predict_file", default=None, type=str, help="json for predictions.")

    parser.add_argument("--max_seq_length", default=512, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--doc_stride", default=450, type=int,
                        help="When splitting up a long document into chunks, how much stride to take between chunks.")
    parser.add_argument("--max_query_length", default=42, type=int,
                        help="The maximum number of tokens for the question. Questions longer than this will "
                             "be truncated to this length.")    
    parser.add_argument("--max_answer_length", default=16, type=int,
                        help="The maximum length of an answer that can be generated. This is needed because the start "
                             "and end predictions are not conditioned on one another.")    
    args = parser.parse_args()

    beam_search_num = 3

    training_modelpath = args.SQG_model
    modelpath = args.bert_model

    
    device = torch.device("cuda")
    
    tokenizer = BertTokenizer.from_pretrained(modelpath)

    model_state_dict = torch.load(training_modelpath)
    print('model load OK')
    model = BertForGenerativeSeq.from_pretrained(modelpath, state_dict=model_state_dict)
    model.eval()
    model.to(device)

    with open(args.predict_file) as file:
        testing_data = json.load(file)

    num = 0
    error_gen = 0
    res = []
    for ele in tqdm(testing_data):
        try:
            context = ele['context']
            question = ele['question']
            answers = ele['answers']
            answer_start = ele['answer_start']

            data = read_data(context = context, answer = answers, answer_start = answer_start)
            features =  convert_data_to_features(
                data=data,
                tokenizer=tokenizer,
                max_seq_length=args.max_seq_length,
                doc_stride=args.doc_stride,
                max_query_length=args.max_query_length)

            input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
            input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
            segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)

            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)

            label_pos = features[0].label_pos
            gen_Qs = []
            step = 0

            output_rank = BS(model, tokenizer, input_ids[0], segment_ids[0], input_mask[0], label_pos, beam_search_num)
            
            while (len(gen_Qs) < beam_search_num):

                tmp_rank = []
                for i in range(beam_search_num - len(gen_Qs)):  
                    label_pos = features[0].label_pos
                    for token_id in output_rank[i][1]:
                        input_ids[0][label_pos] = token_id
                        label_pos += 1

                    input_ids[0][label_pos] = 103 #[MASK]
                    input_mask[0][label_pos] = 1
                    segment_ids[0][label_pos] = 2


                    tmp_rank += BS(model, tokenizer, input_ids[0], segment_ids[0], input_mask[0], label_pos, beam_search_num, output_rank[i])
                
                tmp_rank = sorted(tmp_rank, key=lambda x:x[0], reverse=True)

                output_rank = tmp_rank[:beam_search_num - len(gen_Qs)]
    
                for ele in output_rank[:beam_search_num - len(gen_Qs)]:
                    if '[SEP]' in ele[2]:

                        gen_Qs.append((ele[0]/ele[4], ele[2].replace('[SEP]','').strip()))
                        output_rank.remove(ele)

                if label_pos + 1 >= args.max_seq_length:
                    break 
                                   
            gen_Qs_sort = []
            for ele in sorted(gen_Qs, key=lambda x:x[0], reverse=True):
                gen_Qs_sort.append(ele[1])

            res.append({"context" : context, "question" : question, "answers" : answers, "gen_question" : gen_Qs_sort})
            print({"context" : context, "question" : question, "answers" : answers, "gen_question" : gen_Qs_sort})

            num+=1
            
        except Exception as e:
            error_gen += 1            
            print(context)
            print(question)
            print(answers) 
            print(e)
            raise e


    
    print('num',num)
    print('error',error_gen)
    output_file = os.path.join(args.output_dir, "HLSQG_eval.json")
    json.dump(res, open(output_file, "w"), indent = 4)
    
if __name__ == '__main__':
    main()
