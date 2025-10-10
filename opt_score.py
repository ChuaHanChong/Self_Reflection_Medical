# https://github.com/jinlanfu/GPTScore/blob/main/opt_score.py

import torch


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