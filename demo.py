from src.utils.initialization import read_config_from_yaml
from src.models.model_factory import get_model
from data.reddit import load_data, create_prompts

if __name__ == "__main__":
    

    print("This is demo")
    env = "configs/reddit_llama2_70b.yaml"
    
    cfg = read_config_from_yaml(env)
    
    print('Configuration setup done')
    
    
    
    profiles = load_data(cfg.task_config.path)

    print("Data has been loaded")
    
    data_idx= 102
    
    prompts = create_prompts(profiles[data_idx], cfg.task_config)
    print(prompts[0])

    model = get_model(cfg.gen_model)
    print("Model loaded")   


    max_workers = 1
    results = model.predict_multi(prompts, max_workers=max_workers)
   
    print('------results done-------')
    for res in results:
        print(res[1])
        
        op = res[1]
        
    
