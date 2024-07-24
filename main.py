from src.utils.initialization import read_config_from_yaml
from src.models.model_factory import get_model
from data.reddit import load_data, create_prompts
import argparse
from src.synthetic.synthetic import run_synthetic
from src.configs.config import Task,Experiment

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Process inputs")
    parser.add_argument('--experiment', type=str, default=[Experiment.TOPIC_PRIOR.value], help='The type of experiment')
    # features = income, age, gender, married,
    # features = age, sex, relationship_status, income_level
    parser.add_argument('--feature', type=str, default=['age'], help='Private feature of person')
    parser.add_argument('--hardness', type=list, default=[1,2,3,4,5], help='how hard it was')
    args = parser.parse_args()
    
    env = "configs/reddit_llama2_7b.yaml"
    env = "configs/reddit_llama3_8b.yaml"
    env = "configs/reddit_llama3_70b.yaml"
    # env = "configs/reddit_llama3.1_70b.yaml"
    cfg = read_config_from_yaml(env)
    
    print('Configuration setup done')
    
    if cfg.task == Task.SYNTHETIC:
        # run_synthetic(cfg, Experiment.TOPIC_PRIOR.value, 'sex', args.hardness)
        for exp in args.experiment:
            for feature in args.feature:
                run_synthetic(cfg, exp, feature, args.hardness)
        # run_synthetic(cfg, Experiment.TOPIC_POSTERIOR_SANITIZED.value, args.feature, args.hardness)
        # run_synthetic(cfg, args.experiment, 'married', args.hardness)
        # run_synthetic(cfg, Experiment.TOPIC_POSTERIOR_SANITIZED.value, 'married', args.hardness)
    # profiles = load_data(cfg.task_config.path)

    # print("Data has been loaded")
    
    # data_idx= 102
    
    # prompts = create_prompts(profiles[data_idx], cfg.task_config)
    # print(prompts[0])

    # model = get_model(cfg.gen_model)
    # print("Model loaded")   


    # max_workers = 1
    # results = model.predict_multi(prompts, max_workers=max_workers)
   
    # print('------results done-------')
    # for res in results:
    #     print(res[1])
        
    #     op = res[1]
        
    
