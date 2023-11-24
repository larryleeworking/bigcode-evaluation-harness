# This shell exemplifies how to run the inference for inficoder-eval with this repo
# see detailed instructions in https://infi-coder.github.io/inficoder-eval/

export DATASET_CSV_PATH=..../inficoder-eval-framework/batched_prompts/suite_v2.0.0_dev.csv

# for example, to evaluate Phi-1.5
# first, generate responses
accelerate launch ..../ffqa-evaluation-harness/main.py --model microsoft/phi-1_5 --tasks code-ffqa-v2-phi --batch_size 16 --precision bf16 --n_samples 30 --do_sample True --temperature 0.2 --top_p 0.9 --save_generations --save_references --trust_remote_code --generation_only --max_length_generation 2048 --save_generations_path generations_phi-1_5.json --eos='<|endoftext|>'

# then, join with case names and output a csv file, later the evaluation framework can process
python3 ffqa_processor.py generations_phi-1_5.json references.json ../phi-1_5_output.csv --eos '<|endoftext|>'
