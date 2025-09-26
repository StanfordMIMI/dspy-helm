import dspy
import argparse
import importlib
import os
import random

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=str, required=True)
    parser.add_argument("--optimizer", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--api_base", type=str, default=None)
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--max_bootstrapped_demos", type=int, default=3)
    parser.add_argument("--max_labeled_demos", type=int, default=3)
    parser.add_argument("--num_threads", type=int, default=16)
    parser.add_argument("--val_size", type=str, default=None)
    args = parser.parse_args()

    scenario = getattr(importlib.import_module(f"scenarios"), args.scenario)()
    trainset, valset = scenario.load_data()
    if args.val_size != "" and len(valset) > int(args.val_size):
        valset = random.sample(valset, int(args.val_size))

    if ("o3-mini" in args.model) or ("deepseek-r1" in args.model):
        lm = dspy.LM(model=args.model, api_base=args.api_base, api_key=args.api_key)
        prompt_model = dspy.LM(model="openai/gpt-4o", api_base="", api_key="")
    else:
        lm = dspy.LM(model=args.model, api_base=args.api_base, api_key=args.api_key)
    
    dspy.configure(lm=lm)
    agent = dspy.ChainOfThought("inputs -> output")

    dspy_optimizer = getattr(importlib.import_module("dspy.teleprompt"), args.optimizer)
    
    if (("o3-mini" in args.model) or ("deepseek-r1" in args.model)) and (args.optimizer == "MIPROv2"):
        teleprompter = dspy_optimizer(metric=scenario.metric, max_bootstrapped_demos=args.max_bootstrapped_demos, max_labeled_demos=args.max_labeled_demos, num_threads=args.num_threads, prompt_model=prompt_model)
    else:
        teleprompter = dspy_optimizer(metric=scenario.metric, max_bootstrapped_demos=args.max_bootstrapped_demos, max_labeled_demos=args.max_labeled_demos, num_threads=args.num_threads)
    
    if args.optimizer == "MIPROv2":
        optimized_agent = teleprompter.compile(agent, trainset=trainset, valset=valset, requires_permission_to_run=False)
    else:
        optimized_agent = teleprompter.compile(agent, trainset=trainset, valset=valset)
    
    model_name = args.model.split('/')[-1] if '/' in args.model else args.model
    os.makedirs(f"agents/{args.scenario}/{model_name}", exist_ok=True)
    optimized_agent.save(f"agents/{args.scenario}/{model_name}/{args.optimizer}.json")