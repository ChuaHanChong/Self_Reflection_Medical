import torch
from transformers import GenerationConfig

from evaluate.loop_eval_utils import evaluate_knowledge, evaluate_response


def generate_step(args, model, tokenizer, prompt):
    # print('prompt', prompt)
    inputs = tokenizer(prompt, return_tensors="pt")
    # print('model.device', model.device)
    input_ids = inputs["input_ids"].to(model.device)
    generation_config = GenerationConfig(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        num_beams=args.num_beams,
        early_stopping=True,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    # Without streaming
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            attention_mask=inputs["attention_mask"],
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=args.max_new_tokens,
        )
    s = generation_output.sequences[0][len(input_ids[0]) :]
    output = tokenizer.decode(s)
    return output


def knowledge_loop(args, model, tokenizer, question, knowledge_loop_list=[]):
    print("knowledge_loop")
    THRESHOLD_FACTUAL = args.threshold_fact
    MAX_KNOWLEDGE_LOOP = args.max_knowledge_loop
    candidates = []
    history = []

    prompt = f"Provide background knowledge to answer the given question: \"{question}\"."

    if knowledge_loop_list:
        knowledge = knowledge_loop_list[0]
    else:
        knowledge = generate_step(args, model, tokenizer, prompt)
    # print('==========\n', knowledge)

    loop_i = 0
    if MAX_KNOWLEDGE_LOOP > 1:
        factuality_score = evaluate_knowledge(model, args.demo_num, question, knowledge, tokenizer)
        candidates.append([factuality_score, knowledge])
        history.append([loop_i, knowledge, factuality_score])

    # refine knowledge
    loop_i += 1
    while (loop_i < MAX_KNOWLEDGE_LOOP) and factuality_score < THRESHOLD_FACTUAL:
        if args.no_aspect:
            instruction = f"Please refine the knowledge."
        elif args.no_number:
            instruction = f"The knowledge is not strongly supported by empirical evidence. Please refine the knowledge to improve its factuality."
        else:
            instruction = f"The factuality score for the knowledge is {factuality_score} less than {THRESHOLD_FACTUAL}, which means the knowledge is not strongly supported by empirical evidence. Please refine the knowledge to improve its factuality."

        prompt = f"The provided background knowledge for the question: \"{question}\" is \"{knowledge}\".\n\n{instruction}"
        
        knowledge = generate_step(args, model, tokenizer, prompt)
        # print('==========\n', knowledge)

        factuality_score = evaluate_knowledge(model, args.demo_num, question, knowledge, tokenizer)

        candidates.append([factuality_score, knowledge])
        history.append([loop_i, knowledge, factuality_score])
        loop_i += 1

    if (MAX_KNOWLEDGE_LOOP > 1) and factuality_score < THRESHOLD_FACTUAL:
        # still not satisified, highest_score
        candidates.sort()
        return candidates[-1][-1], history
    else:
        return knowledge, history


def response_loop(args, model, tokenizer, question, final_knowledge, entailment_scorer, ctrleval_scorer):
    print("response_loop")
    THRESHOLD_CONS = args.threshold_consistency
    MAX_RESPONSE_LOOP = args.max_response_loop
    candidates = []
    entailment_score_question_list = []
    history = []

    prompt = f"""Refer to the background knowledge: \"{final_knowledge}\" and answer the question: \"{question}\" with one paragraph."""

    response = generate_step(args, model, tokenizer, prompt)
    loop_i = 0
    if MAX_RESPONSE_LOOP > 1:
        entailment_score_question, cons_score_knowledge = evaluate_response(entailment_scorer, ctrleval_scorer, question, response, final_knowledge)
        candidates.append([(entailment_score_question + cons_score_knowledge) / 2, response])
        entailment_score_question_list.append(entailment_score_question)
        history.append([loop_i, response, entailment_score_question, cons_score_knowledge])

    loop_i += 1
    while loop_i < MAX_RESPONSE_LOOP and cons_score_knowledge < THRESHOLD_CONS:
        if args.no_aspect:
            instruction = f"Please refine the response."
        elif args.no_number:
            instruction = f"The alignment and consistency between response and knowledge are low. Please refine the response to improve its consistency."
        else:
            instruction = f"The consistency score for the response is {cons_score_knowledge} less than {THRESHOLD_CONS}, which means the alignment and consistency between response and knowledge are low. Please refine the response to improve its consistency."

        prompt = f"The generated response for the question: \"{question}\" is \"{response}\" based on the background knowledge: \"{final_knowledge}\".\n\n{instruction}"

        response = generate_step(args, model, tokenizer, prompt)
        # print('==========\n', response)

        entailment_score_question, cons_score_knowledge = evaluate_response(entailment_scorer, ctrleval_scorer, question, response, final_knowledge)
        candidates.append([(entailment_score_question + cons_score_knowledge) / 2, response])
        entailment_score_question_list.append(entailment_score_question)
        history.append([loop_i, response, entailment_score_question, cons_score_knowledge])

        loop_i += 1

    if MAX_RESPONSE_LOOP > 1 and cons_score_knowledge < THRESHOLD_CONS:
        # still not satisified, highest_score
        merge = zip(candidates, entailment_score_question_list)
        merge = sorted(merge)
        candidates, entailment_score_question_list = zip(*merge)
        return candidates[-1][-1], history, entailment_score_question_list[-1] #max
    else:
        return response, history, entailment_score_question


if __name__ == "__main__":
    import argparse
    import os

    import jsonlines
    from tqdm import tqdm
    from transformers import LlamaForCausalLM, LlamaTokenizer

    from CTRLEval.ctrleval import CTRLEval
    from evaluate.sent_similarity import Sent_Similar
    from loop_utils import main_loop

    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=str)
    parser.add_argument("--continue-generate", action="store_true")
    parser.add_argument("--no-number", action="store_true")
    parser.add_argument("--no-aspect", action="store_true")

    parser.add_argument("--out-dir", type=str, default="Alpaca_Lora_7B_loop")
    parser.add_argument("--sources", nargs="+", required=True)
    parser.add_argument("--max-loop", type=int, default=1)
    parser.add_argument("--max-knowledge-loop", type=int, default=1)
    parser.add_argument("--max-response-loop", type=int, default=1)
    parser.add_argument("--demo-num", type=int, default=0)

    parser.add_argument("--threshold-entailment", type=float, default=0.8)
    parser.add_argument("--threshold-fact", type=float, default=-1)
    parser.add_argument("--threshold-consistency", type=float, default=-5)

    parser.add_argument("--max-sample", type=int, default=3000)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1)
    parser.add_argument("--top_k", type=int, default=1)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)

    args = parser.parse_args()

    device = "cuda"

    if args.max_response_loop > 1:
        ctrleval_scorer = CTRLEval(
            iwf_dir="CTRLEval/iwf_full.txt",
            prompt_dir="CTRLEval/prompt/prompt_topic.txt",
            verbal_dir="CTRLEval/prompt/verbal_topic.txt",
            device=device,
        ) #consistency
    # if args.max_knowledge_loop > 1:
    entailment_scorer = Sent_Similar()

    base_model = "meta-llama/Llama-2-7b-hf"

    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    model = LlamaForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map=device,
    )
    model.eval()

    out_dir = f"{args.out_dir}_MaxL{args.max_loop}_MaxKL{args.max_knowledge_loop}MaxRL{args.max_response_loop}_ThE{args.threshold_entailment}ThF{args.threshold_fact}ThC{args.threshold_consistency}_Demo{args.demo_num}"
    os.makedirs(out_dir, exist_ok=True)

    for source in args.sources:
        print(source)
        input_file = args.input_file.format(source=source)
        if args.no_aspect:
            out_file = f"{out_dir}/{source}_T{args.temperature}_no_aspect.jsonl"
        elif args.no_number:
            out_file = f"{out_dir}/{source}_T{args.temperature}_no_number.jsonl"
        else:
            out_file = f"{out_dir}/{source}_T{args.temperature}.jsonl"

        if args.continue_generate and os.path.exists(out_file):
            print("continue generate")
            with jsonlines.open(out_file) as reader:
                old_lines = list(reader)
            with jsonlines.open(input_file) as reader:
                reader = list(reader)
                for i, line in tqdm(enumerate(reader), total=len(reader)):
                    if i < len(old_lines):
                        continue
                    if i > args.max_sample:
                        break
                    final_knowledge, final_response, all_history_knowledge, all_history_response = main_loop(args, line, model, tokenizer, knowledge_loop, response_loop, entailment_scorer, ctrleval_scorer)

                    line.update({"history_knowledge": all_history_knowledge})
                    line.update({"history_response": all_history_response})
                    line.update({"generated_knowledge": final_knowledge})
                    line.update({"generated_answer": final_response})

                    # writer = jsonlines.open(out_file, mode='a')
                    # writer.write(line)
                    # writer.close()

        else:
            with jsonlines.open(input_file) as reader:
                reader = list(reader)
                for i, line in tqdm(enumerate(reader), total=len(reader)):
                    if i > args.max_sample:
                        break
                    final_knowledge, final_response, all_history_knowledge, all_history_response = main_loop(args, line, model, tokenizer, knowledge_loop, response_loop, entailment_scorer, ctrleval_scorer)

                    line.update({"history_knowledge": all_history_knowledge})
                    line.update({"history_response": all_history_response})
                    line.update({"generated_knowledge": final_knowledge})
                    line.update({"generated_answer": final_response})

                    # print(line)
                    # writer = jsonlines.open(out_file, mode='a')
                    # writer.write(line). # FIX: TypeError: Object of type float32 is not JSON serializable
                    # writer.close()
