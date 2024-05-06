from typing import Dict, List

import datasets

from helm.common.hierarchical_logger import hlog
from .scenario import Scenario, Instance, Reference, VALID_SPLIT, CORRECT_TAG, Input, Output


INSTRUCTIONS = {
    "de": "Die folgenden sind Multiple-Choice-Fragen (mit Antworten) über den gesunden Menschenverstand.",
}

INPUT_NOUNS = {
    "de": "Frage",
}

OUTPUT_NOUNS = {
    "de": "Antwort",
}


class HellaSwagScenarioMultilingual(Scenario):
    name = "hellaswag_multilingual"
    description = "Benchmark from https://arxiv.org/pdf/1905.07830.pdf in multiple languages."
    tags = ["knowledge", "multiple_choice"]

    def __init__(self, language: str = "de"):
        super().__init__()
        self.language: str = language

    def download_mmlu(self):
        dataset_dict = datasets.load_dataset("alexandrainst/m_hellaswag", self.language)
        assert isinstance(dataset_dict, datasets.DatasetDict)
        return dataset_dict

    def process_split(self, dataset: datasets.Dataset, split: str) -> List[Instance]:
        instances: List[Instance] = []
        for row in dataset:
            assert isinstance(row, dict)
            question: str = f"{row['activity_label']}: {row['ctx']}"
            answers = row["endings"]
            correct_answer = answers[int(row["label"])]
            assert len(answers) == 4

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
            "val": VALID_SPLIT,
        }

        for split in splits:
            hlog(f"Processing {split} split")
            dataset = dataset_dict[split]
            assert isinstance(dataset, datasets.Dataset)
            instances.extend(self.process_split(dataset, splits[split]))

        return instances
