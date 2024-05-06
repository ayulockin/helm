from typing import Dict, List

import datasets

from helm.common.hierarchical_logger import hlog
from .scenario import Scenario, Instance, Reference, TRAIN_SPLIT, VALID_SPLIT, TEST_SPLIT, CORRECT_TAG, Input, Output


ANSWER_PREFIX = {
    "bn": "উত্তর হল",
    "de": "Die Antwort ist",
    "en": "The answer is",
    "es": "La respuesta es",
    "fr": "La réponse est",
    # TODO: add more languge prefixes as per dataset definition
}


def string_to_bool(input_string):
    # Convert the input string to lowercase to handle case insensitivity
    input_string = input_string.lower()

    if input_string == "true":
        return True
    elif input_string == "false":
        return False
    else:
        raise ValueError("Invalid input: The input must be 'True', 'true', 'False', or 'false'.")


class MGSMScenario(Scenario):
    """Multilingual Grade School Math Benchmark (MGSM) is a benchmark of grade-school math problems,
    proposed in the paper Language models are multilingual chain-of-thought reasoners.

    The same 250 problems from GSM8K are each translated via human annotators in 10 languages. The 10 languages are:
    - Spanish
    - French
    - German
    - Russian
    - Chinese
    - Japanese
    - Thai
    - Swahili
    - Bengali
    - Telugu

    * The train split includes 8 few-shot exemplars that are also manually translated from each language.
    * The test split includes the same 250 problems from GSM8K translated via human annotators in 10 languages.

    Source: https://huggingface.co/datasets/juletxara/mgsm

    If few shot examples (max_train_instances) are used, we will either use the COT prompt or just the final answer.
    For the eval split, we only have the final answer. We will use "The answer is" before the answer.
    """

    name = "mgsm"
    description = "Multilingual Grade School Math Benchmark (MGSM)."
    tags = ["reasoning", "math"]

    def __init__(self, language: str, use_cot_prompt: bool = False):
        super().__init__()
        self.language: str = language
        self.use_cot_prompt: bool = string_to_bool(use_cot_prompt)

    def download_mmlu(self):
        dataset_dict = datasets.load_dataset("juletxara/mgsm", self.language)
        assert isinstance(dataset_dict, datasets.DatasetDict)
        return dataset_dict

    def process_split(self, dataset: datasets.Dataset, split: str) -> List[Instance]:
        instances: List[Instance] = []
        for row in dataset:
            assert isinstance(row, dict)
            question: str = row["question"]
            answer: str = row["answer"]
            answer_number: str = row["answer_number"]

            if split == TRAIN_SPLIT and self.use_cot_prompt:
                reference = Reference(Output(text=answer), tags=[CORRECT_TAG])
            else:
                answer_text = f"{ANSWER_PREFIX.get(self.language, 'The answer is')} {answer_number}."
                reference = Reference(Output(text=answer_text), tags=[CORRECT_TAG])

            instances.append(
                Instance(
                    input=Input(text=question),
                    references=[reference],
                    split=split,
                ),
            )
        return instances

    def get_instances(self, output_path: str) -> List[Instance]:
        # Download the raw data
        dataset_dict = self.download_mmlu()

        # Read all the instances
        instances: List[Instance] = []
        splits: Dict[str, str] = {
            "train": TRAIN_SPLIT,
            "test": TEST_SPLIT,
        }

        for split in splits:
            hlog(f"Processing {split} split")
            dataset = dataset_dict[split]
            assert isinstance(dataset, datasets.Dataset)
            instances.extend(self.process_split(dataset, splits[split]))

        return instances
