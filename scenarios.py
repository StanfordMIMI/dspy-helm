import dspy
import csv
import requests
import os
from sklearn.model_selection import train_test_split
from datasets import load_dataset
from typing import List
from urllib.parse import quote
from bert_score import score
import tempfile

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