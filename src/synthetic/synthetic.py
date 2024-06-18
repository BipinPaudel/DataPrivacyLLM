from data.eval import evaluate, evaluate_age
from data.filter import filter_profiles
from data.reddit import (create_baseline_prompt,create_sanitization_prompt,
                         load_data, load_json_obj_from_file,
                         write_json_lists_to_file)
from src.configs import Config, Experiment
from src.models.model_factory import get_model


def run_synthetic(cfg: Config, experiment, feature, hardness) -> None:
    if experiment == Experiment.BASELINE.value:
        run_baseline_experiment(cfg, feature, hardness)
    elif experiment == Experiment.SANITIZATION.value:
        run_sanitization_experiment(cfg, feature, hardness)
    elif experiment == Experiment.EVALUATION.value:
        run_evaluation_experiment(cfg, feature, hardness)

def run_baseline_experiment(cfg, feature, hardness):
    profiles = load_data(cfg.task_config.path)
    profiles = filter_profiles(profiles, feature, hardness)

    print(len(profiles))
    accurate = 0
    model = get_model(cfg.gen_model)
    print("Model loaded")

    updated_profiles = []
    for profile in profiles:
        # create prompts for baseline
        baseline_prompt = create_baseline_prompt(profile, cfg.task_config)
        print(baseline_prompt[0])
        max_workers = 4

        # temp work here
        results = model.predict_multi(baseline_prompt, max_workers=max_workers)

        for res in results:
            # print(res[1])
            if feature == 'age':
                is_correct, output = evaluate_age(profile, res[1])
            else:
                is_correct, output = evaluate(profile, res[1])
            accurate += is_correct
            profile.parsed_output_baseline = output
            profile.model_response_baseline = res[1]
            updated_profiles.append(profile.to_json())
            print(accurate)
        print(f'Accurate: {accurate}; Total: {len(profiles)}')
        model_name = cfg.gen_model.name.split('/')[1]
        filename = f"results/baseline_{feature}_{''.join(map(str, hardness))}_{model_name}.jsonl"
        write_json_lists_to_file(filename, updated_profiles)

def run_sanitization_experiment(cfg, feature, hardness):
    profiles = load_data(cfg.task_config.path)
    profiles = filter_profiles(profiles, feature, hardness)

    print(len(profiles))
    model = get_model(cfg.gen_model)
    print("Model loaded")
    updated_profiles = []
    for profile in profiles:
        sanitization_prompt = create_sanitization_prompt(profile, cfg.task_config)
        print(sanitization_prompt[0])
        max_workers = 4
        
        # temp work here
        results = model.predict_multi(sanitization_prompt, max_workers=max_workers)
        
        for res in results:
            # print(res[1])
            profile.sanitized_response = res[1]
            updated_profiles.append(profile.to_json())
        filename = f"results/sanitization_{feature}_{''.join(map(str, hardness))}_{cfg.gen_model.name.split('/')[1]}.jsonl"
        write_json_lists_to_file(filename, updated_profiles)
        

def run_evaluation_experiment(cfg, feature, hardness):
    filename = f"results/sanitization_{feature}_{''.join(map(str, hardness))}_{cfg.gen_model.name.split('/')[1]}.jsonl"
    
    # filename = "results/sanitization_income_12345_Meta-Llama-3-70B-Instruct.jsonl"
    sanitized_profiles = load_json_obj_from_file(filename)
    original_profiles = load_data(cfg.task_config.path)
    original_profiles = filter_profiles(original_profiles, feature, hardness)
    original_profiles = add_sanitized_response_to_profile(sanitized_profiles, original_profiles)
    print(len(original_profiles))
    accurate = 0
    model = get_model(cfg.gen_model)
    print("Model loaded")   

    updated_profiles = []
    for profile in original_profiles:
        # create prompts for baseline
        baseline_prompt = create_baseline_prompt(profile, cfg.task_config, use_sanitized_response=True)
        print(baseline_prompt[0])
        max_workers = 4
        
        # temp work here
        results = model.predict_multi(baseline_prompt, max_workers=max_workers)
        
        for res in results:
            # print(res[1])
            if feature == 'age':
                is_correct, output = evaluate_age(profile, res[1])
            else:
                is_correct, output = evaluate(profile, res[1])
            accurate += is_correct
            profile.parsed_output_evaluation = output
            profile.model_response_evaluation = res[1]
            updated_profiles.append(profile.to_json())
            print(accurate)
        print(f'Accurate: {accurate}; Total: {len(original_profiles)}')
        write_filename = f"results/evaluation_{feature}_{''.join(map(str, hardness))}_{cfg.gen_model.name.split('/')[1]}.jsonl"
        write_json_lists_to_file(write_filename, updated_profiles)


def add_sanitized_response_to_profile(sanitized_profiles, original_profiles):
    id_profile_map = {}
    for profile in sanitized_profiles:
        id = profile.get("id")
        id_profile_map[id] = process_sanitized_response(profile.get('sanitized_response'))
    
    for orgi_profile in original_profiles:
        orgi_profile.sanitized_response = id_profile_map[orgi_profile.id]
    return original_profiles


def process_sanitized_response(sanitized_response):
    responses = []
    for response in sanitized_response.split("\n"):
        if response == "" or response == " ":
            continue
        responses.append(response)
    assert responses[-1] is not None
    return responses[-1]