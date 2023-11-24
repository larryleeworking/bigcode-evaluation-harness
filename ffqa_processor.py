import os
import os.path as osp
import argparse
import json
import pandas as pd

# python3 ffqa_processor.py generations_ffqav2_codellama_13b_python.json references.json codellama_13b_python_output.csv

dataset_csv_path = os.environ['DATASET_CSV_PATH']

parser = argparse.ArgumentParser()
parser.add_argument('generation_path', type=str, help='generations .json')
parser.add_argument('references_path', type=str, help='reference .json')
parser.add_argument('out_csv_path', type=str, help='output csv path')
parser.add_argument('--eos', type=str, default='</s>', help='eos token')
if __name__ == '__main__':
    args = parser.parse_args()
    ds = pd.read_csv(dataset_csv_path)
    ds_prompts = {}
    out_completions = []
    out_filenames = []
    for i, row in ds.iterrows():
        ds_prompts[row['filename']] = row['content_prompt']
    with open(args.generation_path, 'r') as f:
        generations = json.load(f)
    with open(args.references_path, 'r') as f:
        references = json.load(f)
    for gens, ref in zip(generations, references):
        print(ref)
        for gen in gens:
            if len(ds_prompts[ref]) > len(gen):
                # print(ref, 'Long:', len(ds_prompts[ref]))
                completion = gen[gen.index(ds_prompts[ref][-len(gen) + 30:]) + len(ds_prompts[ref][-len(gen) + 30:]): ]
            else:
                completion = gen[gen.index(ds_prompts[ref]) + len(ds_prompts[ref]): ]
            
            qwen_start_key = '<|im_start|>assistant\n'
            baichuan_start_key = '<reserved_107>'
            zypher_start_key = '<|assistant|>\n'
            octo_start_key = '\n\nAnswer:'
            qwen_end_key = '<|im_end|>'
            wizard_start_key = '### Response:'
            if qwen_start_key in completion:
                completion = completion[completion.index(qwen_start_key) + len(qwen_start_key): ]
            if baichuan_start_key in completion:
                completion = completion[completion.index(baichuan_start_key) + len(baichuan_start_key): ]
            if zypher_start_key in completion:
                completion = completion[completion.index(zypher_start_key) + len(zypher_start_key): ]
            if octo_start_key in completion:
                completion = completion[completion.index(octo_start_key) + len(octo_start_key): ]
            if wizard_start_key in completion:
                completion = completion[completion.index(wizard_start_key) + len(wizard_start_key): ]
            if qwen_end_key in completion:
                completion = completion[: completion.index(qwen_end_key)]
            
            if completion.count(args.eos) > 0:
                completion = completion[:completion.index(args.eos)]
            out_completions.append(completion)
            out_filenames.append(ref)
    out_df = pd.DataFrame({'filename': out_filenames, 'completion': out_completions})
    out_df.to_csv(args.out_csv_path)
    print(f'Output to {args.out_csv_path}, len={len(out_df)}')
