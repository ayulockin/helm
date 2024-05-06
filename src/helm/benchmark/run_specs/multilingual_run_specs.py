from helm.benchmark.adaptation.adapter_spec import ADAPT_MULTIPLE_CHOICE_JOINT
from helm.benchmark.adaptation.common_adapter_specs import (
    get_multiple_choice_adapter_spec,
    get_generation_adapter_spec,
)
from helm.benchmark.metrics.common_metric_specs import (
    get_basic_generation_metric_specs,
    get_generic_metric_specs,
    get_generative_harms_metric_specs,
    get_exact_match_metric_specs,
)
from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.scenarios.scenario import ScenarioSpec
from helm.benchmark.scenarios.mmlu_scenario_multilingual import (
    INSTRUCTIONS,
    INPUT_NOUNS,
    OUTPUT_NOUNS,
    subject_translations_de,
)
from helm.benchmark.scenarios.hellaswag_scenario_multilingual import (
    INSTRUCTIONS,
    INPUT_NOUNS,
    OUTPUT_NOUNS,
)


@run_spec_function("mmlu_multilingual")
def get_mmlu_spec(subject: str, language: str, method: str = ADAPT_MULTIPLE_CHOICE_JOINT) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.mmlu_scenario_multilingual.MMLUScenarioMultilingual",
        args={"subject": subject, "language": language},
    )

    adapter_spec = get_multiple_choice_adapter_spec(
        method=method,
        instructions=INSTRUCTIONS[language].format(subject_translations_de.get(subject, subject)),
        input_noun=INPUT_NOUNS[language],
        output_noun=OUTPUT_NOUNS[language],
    )

    return RunSpec(
        name=f"mmlu:subject={subject},language={language},method={method}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_exact_match_metric_specs(),
        groups=["mmlu_multilingual"],
    )


@run_spec_function("hellaswag_multilingual")
def get_hellaswag_spec(language: str, method: str = ADAPT_MULTIPLE_CHOICE_JOINT) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.hellaswag_scenario_multilingual.HellaSwagScenarioMultilingual",
        args={"language": language},
    )

    adapter_spec = get_multiple_choice_adapter_spec(
        method=method,
        instructions=INSTRUCTIONS[language],
        input_noun=INPUT_NOUNS[language],
        output_noun=OUTPUT_NOUNS[language],
        max_train_instances=0,  # Since we don't have a train split
    )

    return RunSpec(
        name=f"hellaswag:language={language},method={method}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_exact_match_metric_specs(),
        groups=["hellaswag_multilingual"],
    )


@run_spec_function("mgsm")
def get_mgsm_spec(language: str, use_cot_prompt: bool = False, max_train_instances: int = 5) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.mgsm_scenario_multilingual.MGSMScenario",
        args={"language": language, "use_cot_prompt": use_cot_prompt},
    )

    adapter_spec = get_generation_adapter_spec(
        input_noun="Q",
        output_noun="A",
        max_train_instances=max_train_instances,  # Due to limited context and long example length
        max_tokens=400,  # The paper uses 400 tokens as the max sample length
        stop_sequences=["\n\n"],  # Since answer may contain newlines, we use two as SEP
    )

    return RunSpec(
        name=f"mgsm:language={language},use_cot_prompt={use_cot_prompt},max_train_instances={max_train_instances}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_basic_generation_metric_specs(["exact_match_indicator", "final_number_exact_match"])
        + get_generic_metric_specs()
        + get_generative_harms_metric_specs(),
        groups=["mgsm"],
    )
