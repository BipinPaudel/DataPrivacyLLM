from data.eval import evaluate, evaluate_age, extract_probabilities
from data.filter import filter_profiles, get_unique_private_attribute, get_topics_for_features
from data.reddit import (create_baseline_prompt,create_sanitization_prompt,
                         load_data, load_json_obj_from_file,
                         write_json_lists_to_file,create_topic_prior_prompt,
                         create_topic_posterior_prompt)
from src.configs import Config, Experiment
from src.models.model_factory import get_model
import logging
logging.basicConfig(filename='example.log',
                    filemode='a',  # Append mode, use 'w' for overwrite
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    level=logging.DEBUG)

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
    filename =  f'{feature}_comments.jsonl'
    comments = load_json_obj_from_file(cfg.task_config.path+filename)

    print(len(comments))
    print("Model loaded")
    updated_profiles = []
    prompts = []
    for comment in comments: # CHANGE HERE
        sanitization_prompt = create_sanitization_prompt(comment, feature, cfg.task_config)
        prompts.append(sanitization_prompt[0])
    max_workers = 4
    model = get_model(cfg.gen_model)
    results = model.predict_multi(prompts, max_workers=max_workers)
    results_temp = []
    for res in results:
        print(res[1])
        results_temp.append(res[1])
    for i,comment in enumerate(comments): # CHANGE HERE
        comment["sanitized_response"] = results_temp[i]
        updated_profiles.append(comment)
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


def add_sanitized_response_to_profile(sanitized_comments, original_comments):
    id_comment_map = {}
    for comment in sanitized_comments:
        id = comment.get("id")
        id_comment_map[id] = process_sanitized_response(comment.get('sanitized_response'))
    
    for orgi_profile in original_comments:
        orgi_profile['sanitized_response'] = id_comment_map[orgi_profile['id']]
    return original_comments


def process_sanitized_response(sanitized_response):
    responses = []
    for response in sanitized_response.split("\n"):
        if response == "" or response == " ":
            continue
        responses.append(response)
    assert responses[-1] is not None
    return responses[-1]

def run_topic_prior_experiment(cfg, feature, hardness):
    filename =  f'{feature}_comments.jsonl'
    comments = load_json_obj_from_file(cfg.task_config.path+filename)

    print(len(comments))
    
    private_values = get_unique_private_attribute(comments,feature)
    topics_for_feature = get_topics_for_features(comments)
    topic_values_prob = []

    prompts = []
    for topic in topics_for_feature:
        prompt = create_topic_prior_prompt(topic, (feature, private_values))
        prompts.append(prompt[0])
    model = get_model(cfg.gen_model)
    max_workers=4
    results = model.predict_multi(prompts, max_workers=max_workers)

    topic_counter = 0
    
    #topic prior log
    logging.info(f'{feature}: Private values: {private_values} Topics: {topics_for_feature}')
    
    for res in results:
        print(res[1])
        value_map = extract_probabilities(res[1])
        topic_values_prob.append({'topic': topics_for_feature[topic_counter], 'values': value_map, 'response': res[1]})
        topic_counter += 1
        filename = f"results/topic_prior_{feature}_{''.join(map(str, hardness))}_{cfg.gen_model.name.split('/')[1]}.jsonl"
        write_json_lists_to_file(filename, topic_values_prob)

def run_topic_posterior_experiment(cfg, feature, hardness):
    filename =  f'{feature}_comments.jsonl'
    comments = load_json_obj_from_file(cfg.task_config.path+filename)
    model = get_model(cfg.gen_model) #remove
    
    print("Model loaded")
    private_values = get_unique_private_attribute(comments, feature)
    updated_profiles = []
    prompts = []
    for comment in comments: #CHANGE HERE
        # create prompts for baseline
        baseline_prompt = create_topic_posterior_prompt(comment, (feature, private_values))
        prompts.append(baseline_prompt[0])
    max_workers = 4
    model = get_model(cfg.gen_model)
    # temp work here
    # prompts = prompts[:4] #CHANGE HERE
    results = model.predict_multi(prompts, max_workers=max_workers)
    results_temp = []
    for res in results:
        print(res[1])
        results_temp.append(res[1])
    for i,comment in enumerate(comments): #CHANGE HERE 
        value_map = extract_probabilities(results_temp[i])
        comment['model_response_topic_posterior'] = results_temp[i]
        comment['parsed_topic_posterior'] = value_map
        updated_profiles.append(comment)
    model_name = cfg.gen_model.name.split('/')[1]
    filename = f"results/topic_posterior_{feature}_{''.join(map(str, hardness))}_{model_name}.jsonl"
    write_json_lists_to_file(filename, updated_profiles)
    
def run_topic_posterior_sanitization_experiment(cfg, feature, hardness):
    sanitized_filename = f"results/sanitization_{feature}_{''.join(map(str, hardness))}_{cfg.gen_model.name.split('/')[1]}.jsonl"
    
    # filename = "results/sanitization_income_12345_Meta-Llama-3-70B-Instruct.jsonl"
    sanitized_comments = load_json_obj_from_file(sanitized_filename)
    
    original_filename = f'{feature}_comments.jsonl'
    original_comments = load_json_obj_from_file(cfg.task_config.path+original_filename)
    original_comments = add_sanitized_response_to_profile(sanitized_comments, original_comments) #CHANGE HERE
    print(len(original_comments))
    private_values = get_unique_private_attribute(original_comments, feature)
    
    updated_profiles = []
    prompts = []
    for i,profile in enumerate(original_comments): ##CHANGE HERE
        baseline_prompt = create_topic_posterior_prompt(profile, (feature, private_values), use_sanitized_response=True)
        prompts.append(baseline_prompt[0])
    
    model = get_model(cfg.gen_model)
    max_workers = 4
    results = model.predict_multi(prompts, max_workers=max_workers)
    
    results_temp = []
    for res in results:
        print(res[1])
        results_temp.append(res[1])
        
    for i,comment in enumerate(original_comments): ## CHANGE HERE
        value_map = extract_probabilities(results_temp[i])
        comment['model_response_topic_posterior_sanitized'] = results_temp[i]
        comment['parsed_topic_posterior_sanitized'] = value_map
        updated_profiles.append(comment)
    model_name = cfg.gen_model.name.split('/')[1]
    filename = f"results/topic_posterior_sanitized_{feature}_{''.join(map(str, hardness))}_{model_name}.jsonl"
    write_json_lists_to_file(filename, updated_profiles)
    
    

       