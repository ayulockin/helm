from typing import Dict, List

import datasets

from helm.common.hierarchical_logger import hlog
from .scenario import Scenario, Instance, Reference, TRAIN_SPLIT, VALID_SPLIT, TEST_SPLIT, CORRECT_TAG, Input, Output


LANG_MAP = {
    "de": "DE_DE",
}

INSTRUCTIONS = {
    "de": "Beantworten Sie die folgenden Multiple-Choice-Fragen zu {}. Jede Frage hat vier Antwortmöglichkeiten: A, B, C oder D. Wählen Sie die passendste Antwort und geben Sie nur den entsprechenden Buchstaben an.",
}

INPUT_NOUNS = {
    "de": "Frage",
}

OUTPUT_NOUNS = {
    "de": "Antwort",
}

subject_translations_de = {
    "abstract_algebra": "Abstrakte Algebra",
    "anatomy": "Anatomie",
    "astronomy": "Astronomie",
    "business_ethics": "Wirtschaftsethik",
    "clinical_knowledge": "Klinisches Wissen",
    "college_biology": "Hochschulbiologie",
    "college_chemistry": "Hochschulchemie",
    "college_computer_science": "Hochschulinformatik",
    "college_mathematics": "Hochschulmathematik",
    "college_medicine": "Hochschulmedizin",
    "college_physics": "Hochschulphysik",
    "computer_security": "Computersicherheit",
    "conceptual_physics": "Konzeptuelle Physik",
    "econometrics": "Ökonometrie",
    "electrical_engineering": "Elektrotechnik",
    "elementary_mathematics": "Elementarmathematik",
    "formal_logic": "Formale Logik",
    "global_facts": "Globale Fakten",
    "high_school_biology": "Schulbiologie",
    "high_school_chemistry": "Schulchemie",
    "high_school_computer_science": "Schul-Informatik",
    "high_school_european_history": "Europäische Schulgeschichte",
    "high_school_geography": "Schulgeographie",
    "high_school_government_and_politics": "Schulpädagogik und Politik",
    "high_school_macroeconomics": "Schulmakroökonomie",
    "high_school_mathematics": "Schulmathematik",
    "high_school_microeconomics": "Schulmikroökonomie",
    "high_school_physics": "Schulphysik",
    "high_school_psychology": "Schulpsychologie",
    "high_school_statistics": "Schulstatistik",
    "high_school_us_history": "Amerikanische Schulgeschichte",
    "high_school_world_history": "Weltgeschichte für Schulen",
    "human_aging": "Menschliches Altern",
    "human_sexuality": "Menschliche Sexualität",
    "international_law": "Internationales Recht",
    "jurisprudence": "Rechtswissenschaft",
    "logical_fallacies": "Logische Fehlschlüsse",
    "machine_learning": "Maschinelles Lernen",
    "management": "Management",
    "marketing": "Marketing",
    "medical_genetics": "Medizinische Genetik",
    "miscellaneous": "Verschiedenes",
    "moral_disputes": "Moralische Auseinandersetzungen",
    "moral_scenarios": "Moralische Szenarien",
    "nutrition": "Ernährung",
    "philosophy": "Philosophie",
    "prehistory": "Urgeschichte",
    "professional_accounting": "Professionelles Rechnungswesen",
    "professional_law": "Berufsrecht",
    "professional_medicine": "Berufsmedizin",
    "professional_psychology": "Berufspsychologie",
    "public_relations": "Öffentlichkeitsarbeit",
    "security_studies": "Sicherheitsstudien",
    "sociology": "Soziologie",
    "us_foreign_policy": "US-Außenpolitik",
    "virology": "Virologie",
    "world_religions": "Weltreligionen",
}


class MMLUScenarioMultilingual(Scenario):
    name = "mmlu_multilingual"
    description = "Massive Multitask Language Understanding in Multi lingual language from openai/MMMLU."
    tags = ["knowledge", "multiple_choice"]

    def __init__(self, subject: str, language: str):
        super().__init__()
        self.subject: str = subject
        self.language: str = language

    def download_mmlu(self):
        dataset_dict = datasets.load_dataset("openai/MMMLU", LANG_MAP[self.language])
        assert isinstance(dataset_dict, datasets.DatasetDict)
        return dataset_dict

    def process_split(self, dataset: datasets.Dataset, split: str) -> List[Instance]:
        instances: List[Instance] = []
        for row in dataset:
            assert isinstance(row, dict)
            question: str = row["Question"]
            answers: list[str] = [row["A"], row["B"], row["C"], row["D"]]
            correct_choice: str = row["Answer"]
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
            "test": TEST_SPLIT,
        }

        # Add subject column
        for split in splits:
            dataset = dataset_dict[split]
            assert isinstance(dataset, datasets.Dataset)

        for split in splits:
            hlog(f"Processing {split} split with {self.subject} subject")
            dataset = dataset_dict[split]
            dataset = dataset.filter(lambda x: x["Subject"] == self.subject)
            assert isinstance(dataset, datasets.Dataset)
            instances.extend(self.process_split(dataset, splits[split]))

        return instances
