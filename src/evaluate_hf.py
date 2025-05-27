import os
from huggingface_hub import login
from datasets import load_dataset

from tqdm import tqdm

from eval_utils import compute_results_hf, save_responses


if __name__ == "__main__":
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)    

    # Load the dataset form Hugging Face
    dataset = load_dataset("YanNeu/DASH-B")
    
    # Load the VLM
    
    vlm_name_print = # the model name used for saving results
    ### LOAD YOUR MODEL HERE ###
    
    # Evaluate the dataset
    responses = {}
    for data_dict in tqdm(dataset['test']):
        images = [data_dict['image']]
        q_id = data_dict['question_id']
        prompt = data_dict['question']
        query = f'<image>\n{prompt}'

        ### EVALUATE YOUR MODEL HERE ###
        
        response = None
        responses[q_id] = response

    # Save answers and compute results for DASH-B
    save_responses(responses, output_dir, vlm_name)
    results = compute_results_hf(output_dir, vlm_name, dataset['test'], responses)
    
    print(vlm_name)
    print(results)