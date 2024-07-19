from data.eval import evaluate, evaluate_age, extract_probabilities
from data.filter import filter_profiles, get_unique_private_attribute, get_topics_for_features
from data.reddit import (create_baseline_prompt,create_sanitization_prompt,
                         load_data, load_json_obj_from_file,
                         write_json_lists_to_file,create_topic_prior_prompt,
                         create_topic_posterior_prompt)
from src.configs import Config, Experiment
from src.models.model_factory import get_model


def run_synthetic(cfg: Config, experiment, feature, hardness) -> None:
    if experiment == Experiment.BASELINE.value:
        run_baseline_experiment(cfg, feature, hardness)
    elif experiment == Experiment.SANITIZATION.value:
        run_sanitization_experiment(cfg, feature, hardness)
    elif experiment == Experiment.EVALUATION.value:
        run_evaluation_experiment(cfg, feature, hardness)
    elif experiment == Experiment.TOPIC_PRIOR.value:
        run_topic_prior_experiment(cfg, feature, hardness)
    elif experiment == Experiment.TOPIC_POSTERIOR.value:
        run_topic_posterior_experiment(cfg, feature, hardness)
    elif experiment == Experiment.TOPIC_POSTERIOR_SANITIZED.value:
        run_topic_posterior_sanitization_experiment(cfg, feature, hardness)

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

def run_topic_prior_experiment(cfg, feature, hardness):
    profiles = load_data(cfg.task_config.path)
    profiles = filter_profiles(profiles, feature, hardness)

    print(len(profiles))
    
    private_values = get_unique_private_attribute(profiles,feature)
    topics_for_feature = get_topics_for_features(profiles)
    topic_values_prob = []
    # topics_for_feature = ['Science-Fiction-Reading']
    for topic in topics_for_feature:
        prompt = create_topic_prior_prompt(topic, (feature, private_values))
        model = get_model(cfg.gen_model)
        print(prompt[0])
        max_workers=4
        # temp work here
        results = model.predict_multi(prompt, max_workers=max_workers)

        for res in results:
            value_map = extract_probabilities(res[1])
            topic_values_prob.append({'topic': topic, 'values': value_map, 'response': res[1]})
            
        filename = f"results/topic_prior_{feature}_{''.join(map(str, hardness))}_{cfg.gen_model.name.split('/')[1]}.jsonl"
        write_json_lists_to_file(filename, topic_values_prob)
            
def run_topic_posterior_experiment(cfg, feature, hardness):
    profiles = load_data(cfg.task_config.path)
    profiles = filter_profiles(profiles, feature, hardness)

    print(len(profiles))
    model = get_model(cfg.gen_model)
    print("Model loaded")
    private_values = get_unique_private_attribute(profiles, feature)
    updated_profiles = []
    prompts = []
    for profile in profiles:
        # create prompts for baseline
        baseline_prompt = create_topic_posterior_prompt(profile, (feature, private_values))
        prompts.append(baseline_prompt[0])
        # print(baseline_prompt[0])
    max_workers = 4

    # temp work here
    # prompts = prompts[:2]
    results = model.predict_multi(prompts, max_workers=max_workers)
    results_temp = []
    for res in results:
        print(res[1])
        results_temp.append(res[1])
    for i,profile in enumerate(profiles):
        value_map = extract_probabilities(results_temp[i])
        profile.model_response_topic_posterior = results_temp[i]
        profile.parsed_topic_posterior = value_map
        updated_profiles.append(profile.to_json())
    model_name = cfg.gen_model.name.split('/')[1]
    filename = f"results/topic_posterior_{feature}_{''.join(map(str, hardness))}_{model_name}.jsonl"
    write_json_lists_to_file(filename, updated_profiles)
    
def run_topic_posterior_sanitization_experiment(cfg, feature, hardness):
    filename = f"results/sanitization_{feature}_{''.join(map(str, hardness))}_{cfg.gen_model.name.split('/')[1]}.jsonl"
    
    # filename = "results/sanitization_income_12345_Meta-Llama-3-70B-Instruct.jsonl"
    sanitized_profiles = load_json_obj_from_file(filename)
    original_profiles = load_data(cfg.task_config.path)
    original_profiles = filter_profiles(original_profiles, feature, hardness)
    original_profiles = add_sanitized_response_to_profile(sanitized_profiles, original_profiles)
    print(len(original_profiles))
    private_values = get_unique_private_attribute(original_profiles, feature)
    model = get_model(cfg.gen_model)
    updated_profiles = []
    prompts = []
    for i,profile in enumerate(original_profiles):
        baseline_prompt = create_topic_posterior_prompt(profile, (feature, private_values), use_sanitized_response=True)
        prompts.append(baseline_prompt[0])
    
    max_workers = 4
    results = model.predict_multi(prompts, max_workers=max_workers)
    
    results_temp = []
    for res in results:
        print(res[1])
        results_temp.append(res[1])
        
    for i,profile in enumerate(original_profiles):
        value_map = extract_probabilities(results_temp[i])
        profile.parsed_topic_posterior_sanitized = results_temp[i]
        profile.parsed_topic_posterior_sanitized = value_map
        updated_profiles.append(profile.to_json())
    model_name = cfg.gen_model.name.split('/')[1]
    filename = f"results/topic_posterior_sanitized_{feature}_{''.join(map(str, hardness))}_{model_name}.jsonl"
    write_json_lists_to_file(filename, updated_profiles)
        