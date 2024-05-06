from typing import Dict, List

import datasets

from helm.common.hierarchical_logger import hlog
from .scenario import Scenario, Instance, Reference, TRAIN_SPLIT, VALID_SPLIT, TEST_SPLIT, CORRECT_TAG, Input, Output


INSTRUCTIONS = {
    "de": "Folgende sind Multiple-Choice-Fragen (mit Antworten)",
}

INPUT_NOUNS = {
    "de": "Frage",
}

OUTPUT_NOUNS = {
    "de": "Antwort",
}


class ARCScenarioMultilingual(Scenario):
    """
    This is a machine translated version of ARC dataset in multiple languages.

    Source: https://huggingface.co/datasets/alexandrainst/m_arc
    """

    name = "arc_multilingual"
    description = "AI2 Reasoning Challenge (ARC) in Multiple languages"
    tags = ["knowledge", "multiple_choice"]

    def __init__(self, language: str):
        super().__init__()
        self.language: str = language

    def download_mmlu(self):
        dataset_dict = datasets.load_dataset("alexandrainst/m_arc", self.language)
        assert isinstance(dataset_dict, datasets.DatasetDict)
        return dataset_dict

    def process_split(self, dataset: datasets.Dataset, split: str) -> List[Instance]:
        instances: List[Instance] = []
        for row in dataset:
            if row["option_e"] is not None:
                continue
            assert isinstance(row, dict)
            question: str = row["instruction"]
            answers: list[str] = [row["option_a"], row["option_b"], row["option_c"], row["option_d"]]
            correct_choice: str = row["answer"]
            answers_dict: dict = dict(zip(["A", "B", "C", "D"], answers))
            correct_answer: str = answers_dict[correct_choice]

            def answer_to_reference(answer: str) -> Reference:
                return Reference(Output(text=answer), tags=[CORRECT_TAG] if answer == correct_answer else [])

            instance = Instance(
                input=Input(text=question),
                references=list(map(answer_to_reference, answers)),
                split=split,
            )
            instances.append(instance)

        return instances

    def get_instances(self, output_path: str) -> List[Instance]:
        # Download the raw data
        dataset_dict = self.download_mmlu()

        # Read all the instances
        instances: List[Instance] = []
        splits: Dict[str, str] = {
            "train": TRAIN_SPLIT,
            "val": VALID_SPLIT,
            "test": TEST_SPLIT,
        }

        for split in splits:
            hlog(f"Processing {split} split")
            dataset = dataset_dict[split]
            assert isinstance(dataset, datasets.Dataset)
            instances.extend(self.process_split(dataset, splits[split]))

        return instances
