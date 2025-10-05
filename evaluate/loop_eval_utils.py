#from GPTScore.gpt3_score import gpt3score
#from opt_score import directly_get_score
import torch

def evaluate_response(entailment_scorer, ctrleval_scorer, question, answer, knowledge):
    scores, _ = entailment_scorer.get_scores(question, [answer])
    entailment_score = scores[0]
    
    if knowledge:
        prefix = [knowledge]
        data = [knowledge+'\n'+answer]
        try:
            cons_result = ctrleval_scorer.score(aspect='cons', data=data, prefix=prefix, batch_size=1)
            cons_score = cons_result[0]
        except:
            cons_score = float('-inf')
    else:
        cons_score = float('-inf')
    return entailment_score, cons_score
        
    # print('cosistency', cons_result)
    
#     Pre, Recall, F1 = bert_score.score([response], [golden_response], lang="en", return_hash=False)
#     Pre = Pre.item()
#     Recall = Recall.item()
#     F1 = F1.item()
#     # print('bert_score Pre, Recall, F1', Pre, Recall, F1)

    

def evaluate_knowledge(gptscore_model, demo_num, question, knowledge, gptscore_tokenizer=None):
    PREFIX = {0: f'''Based on Question, please generate the factual knowledge. To do this, please consider these factors: Verifiability, Objectivity, and Reliability of Source. Note that this evaluation should be based on the best available medical knowledge.

Question: {question}
Knowledge: ''',}
    prefix = PREFIX[demo_num]
    
    if gptscore_model == 'gpt3':
        gptscore = gpt3score(input=prefix, output=knowledge,
              gpt3model='davinci003',
              api_key="[YOUR API KEY]")
    else:
        srcs = [prefix]
        tgts = [knowledge]
        score_list = directly_get_score(gptscore_model, gptscore_tokenizer, srcs, tgts, prompt_text="")
        gptscore = score_list[0]
    return gptscore
    


# https://github.com/jinlanfu/GPTScore/blob/main/opt_score.py

def directly_get_score(model, tokenizer, srcs, tgts, prompt_text):
    """ Score a batch of examples """

    device = model.device
    max_length=1024

    def trunk_input(inputs, outputs, reduce_seq, max_length):
        input_ids = tokenizer.encode(inputs)[1:-1]
        output_ids = tokenizer.encode(outputs)[1:-1]
        reduce_seq_ids = tokenizer.encode(reduce_seq)[1:-1]
        total_len = len(input_ids) + len(output_ids)
        if total_len > max_length:
            del_len = len(input_ids) + len(output_ids) - max_length
            reduce_seq_ids = reduce_seq_ids[:len(reduce_seq_ids) - del_len]
            reduce_seq = tokenizer.decode(reduce_seq_ids[1:-1])
        return reduce_seq

    score_list = []
    for i,(src, tgt) in enumerate(zip(srcs, tgts)):
        print('process:'+str(i) + '/'+str(len(srcs)) )
        new_src = trunk_input(src, tgt, src, max_length=max_length)
        src = new_src
        text = src + prompt_text + tgt
        if i <1:
            print('text: ', text)
            print('tgt: ', tgt)
        input_ids = tokenizer.encode(text)
        tgt_ids = tokenizer.encode(tgt)[1:]
        output_ids = [-100] * len(input_ids)
        output_ids[len(input_ids) - len(tgt_ids):] = tgt_ids
        input_ids = torch.LongTensor(input_ids).unsqueeze(0).to(device)
        output_ids = torch.LongTensor(output_ids).unsqueeze(0).to(device)
        try:
            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    labels=output_ids,
                    output_hidden_states=True
                )
            loss, logits, hidden_states = outputs[0], outputs[1], outputs.hidden_states[0]
            loss = loss.item()
            score = -loss
            score_list.append(score)
            print('score: ',score)
        except RuntimeError:
            # traceback.print_exc()
            print('input_ids: ',input_ids)
            print('output_ids: ', output_ids)
            print(f'source: {src}')
            print(f'target: {tgt}')
            # exit(0)
    return score_list