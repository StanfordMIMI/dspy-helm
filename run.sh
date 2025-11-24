scenarios=("mmlu_pro" "gpqa" "gsm8k" "medcalc_bench" "medec" "head_qa" "medbullets")
optimizers=("MIPROv2" "BootstrapFewShotWithRandomSearch")

model=openai/gpt-4o
api_base=""
api_key=""

prompt_model=openai/gpt-4o
prompt_api_base=""
prompt_api_key=""

max_bootstrapped_demos=3
max_labeled_demos=3
num_threads=1

for scenario in "${scenarios[@]}"; do
    for optimizer in "${optimizers[@]}"; do
        echo "Running scenario: $scenario with optimizer: $optimizer"
        val_size=$([ "$optimizer" == "BootstrapFewShotWithRandomSearch" ] && echo "100" || echo "")
        python main.py \
            --scenario $scenario \
            --optimizer $optimizer \
            --model $model \
            --api_base "$api_base" \
            --api_key "$api_key" \
            --prompt_model $prompt_model \
            --prompt_api_base "$prompt_api_base" \
            --prompt_api_key "$prompt_api_key" \
            --max_bootstrapped_demos $max_bootstrapped_demos \
            --max_labeled_demos $max_labeled_demos \
            --num_threads $num_threads \
            --val_size "$val_size"
    done
done
