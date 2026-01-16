import dspy
import csv
import requests
import os
from sklearn.model_selection import train_test_split
from datasets import load_dataset
from datasets import Dataset
from typing import List, Dict, Optional, Tuple
from urllib.parse import quote
import tempfile
import re
import random
import json

def to_example(x):
    return dspy.Example(**x).with_inputs("inputs")

class medcalc_bench:
    def __init__(self, test_size=0.1, seed=42):
        self.test_size = test_size
        self.seed = seed

    @staticmethod
    def make_prompt(row):
        return (
            "Given a patient note and a clinical question, compute the requested medical value.\n\n"
            f"Patient note: {row['Patient Note']}\n\n"
            f"Question: {row['Question']}\n\n"
            "Answer only the requested quantity without units. No explanation needed:"
        )

    @staticmethod
    def metric(example, pred, trace=None):
        example["answer"] = example["output"]
        pred["answer"] = pred["output"]
        return dspy.evaluate.metrics.answer_exact_match(example, pred, trace)

    @staticmethod
    def metric_with_feedback(example, pred, trace=None, pred_name=None, pred_trace=None):
        score = medcalc_bench.metric(example, pred, trace)
        expected_answer = example["output"]
        predicted_answer = pred["output"]
        
        if score == 1.0:
            feedback = f"You correctly computed the medical value as '{expected_answer}'. Your calculation is accurate."
        else:
            feedback = f"You incorrectly computed the medical value as '{predicted_answer}'. The correct answer is '{expected_answer}'. Review your medical calculation carefully."
        
        return dspy.Prediction(score=score, feedback=feedback)
        
    def load_data(self):
        dataset = load_dataset("ncbi/MedCalc-Bench-v1.0")
        split = dataset["train"].train_test_split(test_size=self.test_size, seed=self.seed)
        train_examples = [
            {"inputs": self.make_prompt(row), "output": row["Ground Truth Answer"]}
            for row in split["train"]
        ]
        val_examples = [
            {"inputs": self.make_prompt(row), "output": row["Ground Truth Answer"]}
            for row in split["test"]
        ]
        trainset = [to_example(x) for x in train_examples]
        valset = [to_example(x) for x in val_examples]
        return trainset, valset
    
class medec:
    def __init__(self, test_size=0.1, seed=42):
        self.test_size = test_size
        self.seed = seed

    @staticmethod
    def make_prompt(row):
        return (
            "The following is a medical narrative about a patient. "
            "You are a skilled medical doctor reviewing the clinical text. "
            "The text is either correct or contains one error. "
            "The text has a sentence per line. Each line starts with the sentence ID, followed by a space character then the sentence to check. "
            "Check every sentence of the text. "
            "If the text is correct return the following output: CORRECT. "
            "If the text has a medical error, return the sentence ID of the sentence containing the error, followed by a space, and a corrected version of the sentence.\n\n"
            f"Clinical Note: {row['Sentences']}\n\n"
            "Answer:"
        )

    @staticmethod
    def metric(example, pred, trace=None):
        example["answer"] = example["output"]
        pred["answer"] = pred["output"]
        return dspy.evaluate.metrics.answer_exact_match(example, pred, trace)

    @staticmethod
    def metric_with_feedback(example, pred, trace=None, pred_name=None, pred_trace=None):
        score = medec.metric(example, pred, trace)
        expected_answer = example["output"]
        predicted_answer = pred["output"]
        
        if score == 1.0:
            if expected_answer == "CORRECT":
                feedback = "You correctly identified that the clinical text is error-free."
            else:
                feedback = f"You correctly identified the medical error and provided the correction: '{expected_answer}'."
        else:
            if expected_answer == "CORRECT":
                feedback = f"You incorrectly identified an error in the text. The text is actually correct. Your response: '{predicted_answer}' should have been 'CORRECT'."
            else:
                feedback = f"You failed to identify the medical error correctly. The correct response should be: '{expected_answer}'. Your response: '{predicted_answer}'."
        
        return dspy.Prediction(score=score, feedback=feedback)

    def _download_csv(self):
        local_path = "MEDEC-Full-TrainingSet-with-ErrorType.csv"
        if not os.path.exists(local_path):
            print(f"Downloading MEDEC training set to {local_path} ...")
            r = requests.get("https://raw.githubusercontent.com/abachaa/MEDEC/49c59dcba43a7af8717590a405b11a57f42b04df/MEDEC-MS/MEDEC-Full-TrainingSet-with-ErrorType.csv")
            r.raise_for_status()
            with open(local_path, "w", encoding="utf-8") as f:
                f.write(r.text)
        return local_path

    def _get_answer(self, row):
        if int(row.get("Error Flag", 0)) == 1 and row.get("Corrected Sentence", "").strip() != "NA" and row.get("Error Sentence ID", "-1").strip() != "-1":
            return f"{row['Error Sentence ID'].strip()} {row['Corrected Sentence'].strip()}"
        else:
            return "CORRECT"

    def load_data(self):
        csv_path = self._download_csv()
        rows = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if not row.get("Sentences"):
                    continue
                rows.append(row)
        os.remove(csv_path)

        train_rows, val_rows = train_test_split(rows, test_size=self.test_size, random_state=self.seed)

        train_examples = [
            {"inputs": self.make_prompt(row), "output": self._get_answer(row)}
            for row in train_rows
        ]
        val_examples = [
            {"inputs": self.make_prompt(row), "output": self._get_answer(row)}
            for row in val_rows
        ]
        trainset = [to_example(x) for x in train_examples]
        valset = [to_example(x) for x in val_examples]
        return trainset, valset

class head_qa:
    def __init__(self, test_size=0.1, seed=42):
        self.test_size = test_size
        self.seed = seed

    @staticmethod
    def make_prompt(row):
        question = row["qtext"]
        options = "\n".join([f"{chr(65 + i)}. {option['atext']}" for i, option in enumerate(row["answers"])])
        return (
            "You are a highly knowledgeable AI assistant specializing in biomedical sciences. Your task is to answer "
            "multiple-choice questions accurately based on the options provided. Each question will relate to biomedical concepts, "
            "and you will be asked to choose the most appropriate answer.\n\n"
            "Select the correct answer by outputting only the letter corresponding to your choice (A, B, C, or D).\n\n"
            f"Question: {question}\n{options}\nAnswer:"
        )

    @staticmethod
    def get_answer_letter(row):
        for idx, option in enumerate(row["answers"]):
            if str(option["aid"]) == str(row["ra"]):
                return chr(65 + idx)
        return None

    @staticmethod
    def metric(example, pred, trace=None):
        example["answer"] = example["output"]
        pred["answer"] = pred["output"]
        return dspy.evaluate.metrics.answer_exact_match(example, pred, trace)

    @staticmethod
    def metric_with_feedback(example, pred, trace=None, pred_name=None, pred_trace=None):
        score = head_qa.metric(example, pred, trace)
        expected_answer = example["output"]
        predicted_answer = pred["output"]
        
        if score == 1.0:
            feedback = f"You correctly selected answer '{expected_answer}' for the biomedical question."
        else:
            feedback = f"You incorrectly selected answer '{predicted_answer}'. The correct answer is '{expected_answer}'. Review the biomedical question and options carefully."

        return dspy.Prediction(score=score, feedback=feedback)

    def load_data(self):
        dataset = load_dataset("dvilares/head_qa", "en")
        split = dataset["train"].train_test_split(test_size=self.test_size, seed=self.seed)
        train_examples = [
            {"inputs": self.make_prompt(row), "output": self.get_answer_letter(row)}
            for row in split["train"]
        ]
        val_examples = [
            {"inputs": self.make_prompt(row), "output": self.get_answer_letter(row)}
            for row in split["test"]
        ]
        trainset = [to_example(x) for x in train_examples]
        valset = [to_example(x) for x in val_examples]
        return trainset, valset

class medbullets:
    DATASET_DOWNLOAD_URL = "https://raw.githubusercontent.com/HanjieChen/ChallengeClinicalQA/refs/heads/main/medbullets/medbullets_op4.csv"
    POSSIBLE_ANSWER_CHOICES: List[str] = ["A", "B", "C", "D", "E"]

    def __init__(self, test_size=0.2, seed=42):
        self.test_size = test_size
        self.seed = seed

    @staticmethod
    def make_prompt(row):
        question = row["question"]
        options = []
        for i, letter in enumerate(medbullets.POSSIBLE_ANSWER_CHOICES):
            key = f'op{chr(97 + i)}'
            if key in row and row[key].strip():
                options.append(f"{letter}. {row[key]}")
        options_str = "\n".join(options)
        return (
            "You are a highly knowledgeable AI assistant specializing in medicine. "
            "Your task is to answer medical questions similar to those found on the USMLE Step 2/3 exams. "
            "You will be provided with a clinical scenario followed by several multiple-choice options.\n\n"
            "Select the correct answer by outputting only the letter corresponding to your choice (A, B, C, D, or E).\n\n"
            f"Clinical Scenario: {question}\n{options_str}\nAnswer:"
        )

    @staticmethod
    def metric(example, pred, trace=None):
        example["answer"] = example["output"]
        pred["answer"] = pred["output"]
        return dspy.evaluate.metrics.answer_exact_match(example, pred, trace)

    @staticmethod
    def metric_with_feedback(example, pred, trace=None, pred_name=None, pred_trace=None):
        score = medbullets.metric(example, pred, trace)
        expected_answer = example["output"]
        predicted_answer = pred["output"]
        
        if score == 1.0:
            feedback = f"You correctly selected answer '{expected_answer}' for the USMLE-style medical question."
        else:
            feedback = f"You incorrectly selected answer '{predicted_answer}'. The correct answer is '{expected_answer}'. Review the clinical scenario and medical options carefully."

        return dspy.Prediction(score=score, feedback=feedback)

    def _download_csv(self):
        local_path = "medbullets_op4.csv"
        if not os.path.exists(local_path):
            print(f"Downloading MedBullets training set to {local_path} ...")
            r = requests.get(self.DATASET_DOWNLOAD_URL)
            r.raise_for_status()
            with open(local_path, "w", encoding="utf-8") as f:
                f.write(r.text)
        return local_path

    def process_csv(self, csv_path: str) -> List[dict]:
        data_rows = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if not row.get("question") or not row.get("answer_idx"):
                    print(f"Skipping invalid row: {row}")
                    continue
                correct_option = row["answer_idx"]
                formatted_row = {
                    "inputs": self.make_prompt(row),
                    "output": correct_option
                }
                data_rows.append(formatted_row)
        return data_rows

    def load_data(self):
        csv_path = self._download_csv()
        all_data = self.process_csv(csv_path)
        os.remove(csv_path)
        train_data, val_data = train_test_split(all_data, test_size=self.test_size, random_state=self.seed)
        train_set = [to_example(data) for data in train_data]
        val_set = [to_example(data) for data in val_data]
        return train_set, val_set

class mmlu_pro:
    LABELS = ["A","B","C","D","E","F","G","H","I","J"]

    def __init__(self, test_size=0.5, seed=42):
        self.subject =  "all"
        self.use_chain_of_thought = True
        self.revision = "3373e0b"
        self.test_size = test_size
        self.seed = seed

    @staticmethod
    def _helm_input_prefix() -> str:
        return "What is the correct answer to this question: "

    @staticmethod
    def _helm_input_suffix() -> str:
        return "\nChoices:\n"

    @staticmethod
    def _helm_global_suffix_cot() -> str:
        return (
            "Let’s think step by step. Based on your reasoning, what is the single, "
            "most likely answer choice? Format your response as follows: "
            "\"The correct answer is (insert answer here)\"."
        )

    @staticmethod
    def _helm_global_suffix_nocot() -> str:
        return 'Format your response as follows: "The correct answer is (insert answer here)".'

    @staticmethod
    def _final_uppercase_letter_instruction() -> str:
        return (
            'In your response, replace "insert answer here" with the single uppercase letter '
            "corresponding to your answer."
        )

    @classmethod
    def _choices_block(self, options: List[str]) -> str:
        assert len(options) == 10, f"Expected 10 options, got {len(options)}"
        return "\n".join([f"{label}. {opt}" for label, opt in zip(self.LABELS, options)])

    def make_prompt(self, row: Dict) -> str:
        question = row["question"]
        options = row["options"]

        prefix = self._helm_input_prefix()
        suffix = self._helm_input_suffix()
        global_suffix = (self._helm_global_suffix_cot() if self.use_chain_of_thought else self._helm_global_suffix_nocot())
        final_inst = self._final_uppercase_letter_instruction()

        prompt = (
            f"{prefix}{question} \n"
            f"{suffix}"
            f"{self._choices_block(options)} \n"
            f"{global_suffix} \n\n"
            f"{final_inst}"
        )
        return prompt

    @staticmethod
    def _extract_letter(text: str) -> Optional[str]:
        if not text:
            return None
        m = re.search(r"\b([A-J])\b", text.upper())
        if not m:
            m = re.search(r"\(([A-J])\)", text.upper())
        return m.group(1) if m else None

    @staticmethod
    def metric(example, pred, trace=None):
        example["answer"] = str(example["output"]).strip().upper()
        pred["answer"] = mmlu_pro._extract_letter(pred.get("output", "")) or ""
        return dspy.evaluate.metrics.answer_exact_match(example, pred, trace)

    @staticmethod
    def metric_with_feedback(example, pred, trace=None, pred_name=None, pred_trace=None):
        score = mmlu_pro.metric(example, pred, trace)
        expected_answer = str(example["output"]).strip().upper()
        predicted_answer = pred["output"]
        
        if score == 1.0:
            feedback = f"You correctly identified the answer as '{expected_answer}'. Your response properly follows the required format and contains the correct answer letter."
        else:
            feedback = f"You incorrectly identified the answer by outputting '{predicted_answer}'.\n\nThe correct answer is '{expected_answer}'. Review the question and options carefully to identify the right answer."
        
        return dspy.Prediction(score=score, feedback=feedback)

    def load_data(self):
        ds_all = load_dataset("TIGER-Lab/MMLU-Pro", revision=self.revision)
        ds = ds_all["validation"]
        split = ds.train_test_split(test_size=self.test_size, seed=self.seed)

        def row_to_example(row: Dict) -> Dict:
            return {"inputs": self.make_prompt(row), "output": str(row["answer"]).strip().upper()}

        train_examples = [row_to_example(r) for r in split["train"]]
        val_examples = [row_to_example(r) for r in split["test"]]

        trainset = [to_example(x) for x in train_examples]
        valset = [to_example(x) for x in val_examples]
        return trainset, valset

class gpqa:
    LABELS: List[str] = ["A", "B", "C", "D"]
    
    def __init__(self, test_size=0.2, seed=42):
        self.subset = "gpqa_main"
        self.use_chain_of_thought = True
        self.revision = "90b8e5be2b1d3d2dbfe016cdab47981150600c4a"
        self.include_final_letter_instruction = True
        self.test_size = test_size
        self.seed = seed
        self._rng = random.Random(self.seed)

    @staticmethod
    def _helm_input_prefix() -> str:
        return "What is the correct answer to this question: "

    @staticmethod
    def _helm_input_suffix() -> str:
        return "\nChoices: \n"

    @staticmethod
    def _helm_global_suffix_cot() -> str:
        return (
            "Let’s think step by step. Based on your reasoning, what is the single, "
            "most likely answer choice? Format your response as follows: "
            "\"The correct answer is (insert answer here)\"."
        )

    @staticmethod
    def _helm_global_suffix_nocot() -> str:
        return 'Format your response as follows: "The correct answer is (insert answer here)".'

    @staticmethod
    def _final_uppercase_letter_instruction() -> str:
        return (
            'In your response, replace "insert answer here" with the single uppercase letter '
            "corresponding to your answer."
        )

    @classmethod
    def _choices_block(self, shuffled_opts: List[Tuple[str, bool]]) -> str:
        assert len(shuffled_opts) == 4, f"Expected 4 options, got {len(shuffled_opts)}"
        lines = []
        for label, (opt_text, _) in zip(self.LABELS, shuffled_opts):
            lines.append(f"({label}) {opt_text}")
        return "\n".join(lines)

    def _shuffle_and_label(self, row: Dict) -> Tuple[List[Tuple[str, bool]], str]:
        options = [(row["Correct Answer"].strip(), True), (row["Incorrect Answer 1"].strip(), False), (row["Incorrect Answer 2"].strip(), False), (row["Incorrect Answer 3"].strip(), False)]
        self._rng.shuffle(options)
        correct_idx = next(i for i, (_, is_correct) in enumerate(options) if is_correct)
        gold_letter = self.LABELS[correct_idx]
        return options, gold_letter

    def make_prompt(self, row: Dict) -> Tuple[str, str]:
        question = row["Question"].strip()
        shuffled_opts, gold_letter = self._shuffle_and_label(row)
        prefix = self._helm_input_prefix()
        suffix = self._helm_input_suffix()
        global_suffix = (self._helm_global_suffix_cot() if self.use_chain_of_thought else self._helm_global_suffix_nocot())

        prompt = (
            f"{prefix}{question} \n"
            f"{suffix}"
            f"{self._choices_block(shuffled_opts)} \n"
            f"{global_suffix}"
        )
        if self.include_final_letter_instruction:
            prompt += "\n\n" + self._final_uppercase_letter_instruction()

        return prompt, gold_letter

    @staticmethod
    def _extract_letter(text: str) -> Optional[str]:
        if not text:
            return None
        t = str(text).upper()
        m = re.search(r"\b([A-D])\b", t)
        if not m:
            m = re.search(r"\(([A-D])\)", t)
        return m.group(1) if m else None

    @staticmethod
    def metric(example, pred, trace=None):
        example["answer"] = str(example["output"]).strip().upper()
        pred["answer"] = gpqa._extract_letter(pred.get("output", "")) or ""
        return dspy.evaluate.metrics.answer_exact_match(example, pred, trace)

    @staticmethod
    def metric_with_feedback(example, pred, trace=None, pred_name=None, pred_trace=None):
        score = gpqa.metric(example, pred, trace)
        expected_answer = str(example["output"]).strip().upper()
        predicted_answer = pred["output"]
        
        if score == 1.0:
            feedback = f"You correctly selected answer '{expected_answer}' for the graduate-level question."
        else:
            feedback = f"You incorrectly answered the question by outputting '{predicted_answer}'. The correct answer is '{expected_answer}'. Review the complex question and options carefully."

        return dspy.Prediction(score=score, feedback=feedback)

    def load_data(self):
        ds = load_dataset("Idavidrein/gpqa", self.subset, revision=self.revision, split="train")
        split = ds.train_test_split(test_size=self.test_size, seed=self.seed)

        def row_to_example(row: Dict) -> Dict:
            prompt, gold_letter = self.make_prompt(row)
            return {"inputs": prompt, "output": gold_letter}

        train_examples = [row_to_example(r) for r in split["train"]]
        val_examples = [row_to_example(r) for r in split["test"]]

        trainset = [to_example(x) for x in train_examples]
        valset = [to_example(x) for x in val_examples]
        return trainset, valset

class gsm8k:
    def __init__(self, test_size=0.1, seed=42):
        self.base_url = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data"
        self.test_size = test_size
        self.seed = seed

    @staticmethod
    def make_prompt(row):
        return f"Q: {row['question'].strip()}\nA:"

    @staticmethod
    def _extract_final_number(text: str) -> str:
        if text is None:
            return ""
        match = re.search(r"The answer is\s*([-+]?\d+)", text)
        if match:
            return match.group(1)
        numbers = re.findall(r"[-+]?\d+", text)
        return numbers[-1] if numbers else ""

    @staticmethod
    def metric(example, pred, trace=None):
        example["answer"] = str(example["output"]).strip()
        pred["answer"] = gsm8k._extract_final_number(pred.get("output", ""))
        return dspy.evaluate.metrics.answer_exact_match(example, pred, trace)

    @staticmethod
    def metric_with_feedback(example, pred, trace=None, pred_name=None, pred_trace=None):
        score = gsm8k.metric(example, pred, trace)
        expected_answer = str(example["output"]).strip()
        predicted_answer = gsm8k._extract_final_number(pred.get("output", ""))
        full_solution = example["solution"]
        
        if score == 1.0:
            feedback = f"You correctly solved the math problem. The answer is '{expected_answer}'."
        else:
            feedback = f"You incorrectly calculated the answer as '{predicted_answer}'. The correct answer is '{expected_answer}'.\n\nFull solution: {full_solution}\n\nReview your math problem solving steps carefully and follow the proper reasoning approach."

        return dspy.Prediction(score=score, feedback=feedback)

    def _load_split(self, split_name):
        cache_dir = ".cache"
        os.makedirs(cache_dir, exist_ok=True)
        local_path = os.path.join(cache_dir, f"gsm8k_{split_name}.jsonl")

        if not os.path.exists(local_path):
            url = f"{self.base_url}/{split_name}.jsonl"
            resp = requests.get(url)
            resp.raise_for_status()
            with open(local_path, "w") as f:
                f.write(resp.text)

        with open(local_path, "r") as f:
            data = [json.loads(line) for line in f]
        
        os.remove(local_path)
        if not os.listdir(cache_dir):
            os.rmdir(cache_dir)
        
        return data

    def load_data(self):
        dataset = self._load_split("train")
        ds = Dataset.from_list(dataset)
        split = ds.train_test_split(test_size=self.test_size, seed=self.seed)

        train_examples = [{"inputs": self.make_prompt(row), "output": self._extract_final_number(row["answer"]), "solution": row["answer"]} for row in split["train"]]
        val_examples = [{"inputs": self.make_prompt(row), "output": self._extract_final_number(row["answer"]), "solution": row["answer"]} for row in split["test"]]

        trainset = [to_example(x) for x in train_examples]
        valset = [to_example(x) for x in val_examples]
        return trainset, valset

class wildbench:
    def __init__(self, test_size=0.1, seed=42, subset="v1-legacy"):
        self.subset = subset
        self.test_size = test_size
        self.seed = seed
        self.revision = "7c05c1b4550282b2ed6a2e6ac5db069f1e07df5c"
        self.return_feedback = False
    
    def make_prompt_with_checklist(self, row: Dict) -> tuple:
        conversation = row["conversation_input"]
        prompt_parts = []
        for message in conversation:
            role = message["role"]
            content = message["content"]
            prompt_parts.append(f"{role}\n{content}")
        
        prompt = "\n\n".join(prompt_parts)
        checklist = row.get("checklist", [])
        return prompt, checklist
    
    def metric(self, example, pred, trace=None):
        SCORE_TEMPLATE = """
        # Instruction 

        You are an expert evaluator. Your task is to evaluate the quality of the responses generated by AI models. 
        We will provide you with the user query and an AI-generated responses.
        You should first read the user query and the conversation history carefully for analyzing the task, and then evaluate the quality of the responses based on and rules provided below.

        # Conversation between User and AI

        ## History
        <|begin_of_history|>

        {$history}

        <|end_of_history|> 

        ## Current User Query
        <|begin_of_query|>

        {$user_query}

        <|end_of_query|>

        ## AI Response
        <|begin_of_response|>

        {$model_output}

        <|end_of_response|>
        

        # Evaluation   

        ## Checklist 

        <|begin_of_checklist|>

        {$checklist}

        <|end_of_checklist|>

        Please use this checklist to guide your evaluation, but do not limit your assessment to the checklist.

        ## Rules 

        You should compare the above response based on your analysis of the user queries and the conversation history.
        You should first write down your analysis and the checklist that you used for the evaluation, and then provide your assessment according to the checklist.
        The scores are in the range of 1~10, where 1 means the response is very poor and 10 means the response is perfect.
        Here are more detailed criteria for the scores:

        - Score 1~2: The response is very poor and does not make sense at all.
        - Score 3~4: The response is poor and does help user solve the problem in a meaningful way.
        - Score 5~6: The response is fair but has some issues (e.g., factual errors, hallucinations, missing key information).
        - Score 7~8: The response is good enough but could be improved in some ways.
        - Score 9~10: The response is perfect and provides helpful information that can help user solve the problem.

        ## Output Format 
        First, please output your analysis for the model response, and then summarize your assessment to two aspects: "strengths" and "weaknesses"; Finally, please write down your rating for the assessment.

        Please provide your evaluation results in the following json format by filling in the placeholders in []:
        ```
        {
            "strengths": "[analysis for the strengths of the response]",
            "weaknesses": "[analysis for the weaknesses of the response]",
            "score": "[1~10]"
        }
        ```
        """
        
        pattern = re.compile(r'"strengths"\s*:\s*"(.*?)"\s*,\s*"weaknesses"\s*:\s*"(.*?)"\s*,\s*"score"\s*:\s*(".*?"|\d+)', re.DOTALL)
        
        def _extract_score_from_response(response_text: str) -> Optional[float]:
            response_parts = pattern.search(response_text)
            if not response_parts:
                return None
            
            score_text = response_parts[3].strip().strip('"')
            try:
                return float(score_text)
            except ValueError:
                return None
        
        input_text = example['inputs']
        checklist = example['checklist']
        
        lines = input_text.split('\n')
        conversation = []
        current_role = None
        current_content = []
        
        for line in lines:
            if line in ['user', 'assistant']:
                if current_role is not None:
                    conversation.append({"role": current_role, "content": '\n'.join(current_content)})
                current_role = line
                current_content = []
            else:
                current_content.append(line)
        
        if current_role is not None:
            conversation.append({"role": current_role, "content": '\n'.join(current_content)})
        model_output = pred.get('output', '')
        
        history = []
        for round in conversation[:-1]:
            noun = "USER: " if round["role"] == "user" else "ASSISTANT: "
            history.append(noun + round["content"])
        history_text = "\n\n".join(history)
        
        user_query_text = conversation[-1]["content"]
        checklist_text = "\n".join([f"- {checklist_item}" for checklist_item in checklist])
        eval_prompt = (SCORE_TEMPLATE.replace("{$history}", history_text).replace("{$user_query}", user_query_text).replace("{$model_output}", model_output).replace("{$checklist}", checklist_text))
        
        try:
            from openai import OpenAI
            client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
            
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an expert evaluator for AI model responses."},
                    {"role": "user", "content": eval_prompt}
                ],
                temperature=0.0
            )
            
            response_text = response.choices[0].message.content
            score = _extract_score_from_response(response_text)
            
            if score is None:
                print(f"Error calling OpenAI API for evaluation: {response_text}")
                return 0.0 if not self.return_feedback else (0.0, "")
            
            if self.return_feedback:
                return (score-1.0)/9.0, response_text
            else:
                return score >= 8.0 if trace is not None else (score-1.0)/9.0
            
        except Exception as e:
            print(f"Error calling OpenAI API for evaluation: {e}")
            return 0.0 if not self.return_feedback else (0.0, "")

    def metric_with_feedback(self, example, pred, trace=None, pred_name=None, pred_trace=None):
        self.return_feedback = True
        score, feedback = self.metric(example, pred, trace)
        return dspy.Prediction(score=score, feedback=feedback)
    
    def load_data(self):
        ds = load_dataset("allenai/WildBench", self.subset, split="test", revision=self.revision)
        split = ds.train_test_split(test_size=self.test_size, seed=self.seed)
        
        def row_to_example(row: Dict) -> Dict:
            prompt, checklist = self.make_prompt_with_checklist(row)
            return {"inputs": prompt, "output": "", "checklist": checklist}
        
        train_examples = [row_to_example(r) for r in split["train"]]
        val_examples = [row_to_example(r) for r in split["test"]]
        
        trainset = [to_example(x) for x in train_examples]
        valset = [to_example(x) for x in val_examples]
        return trainset, valset