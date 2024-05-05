from typing import Dict, List

import datasets

from helm.common.hierarchical_logger import hlog
from .scenario import Scenario, Instance, Reference, TRAIN_SPLIT, VALID_SPLIT, TEST_SPLIT, CORRECT_TAG, Input, Output


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
