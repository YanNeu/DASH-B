import os
import json

IMG_DIR_NEG = 'images/neg'
IMG_DIR_POS = 'images/pos'
JSON_NEG = 'images/dash_benchmark_neg.json'
JSON_POS = 'images/dash_benchmark_pos.json'


def load_json(filename):
    with open(filename, 'r') as f:
        return json.load(f)


def get_prompt(target_label, prompt_id=4):
    if prompt_id == 0:
        prompt = f'Does this image contain a {target_label}? Please answer only with yes or no.'
    elif prompt_id == 1:
        prompt = f'Is {target_label} in the image? Please answer only with yes or no.'
    elif prompt_id == 2:
        prompt = f'Is {target_label} visible in the image? Please answer only with yes or no.'
    elif prompt_id == 3:
        prompt = f'Does the image show a {target_label}? Please answer only with yes or no.'
    elif prompt_id == 4:
        prompt = f'Can you see a {target_label} in this image? Please answer only with yes or no.'
    elif prompt_id == 5:
        prompt = f'Is there a {target_label} present in the image? Please answer only with yes or no.'
    elif prompt_id == 6:
        prompt = f'Does this picture include a {target_label}? Please answer only with yes or no.'
    elif prompt_id == 7:
        prompt = f'Is a {target_label} depicted in this image? Please answer only with yes or no.'
    elif prompt_id == 8:
        prompt = f'Is a {target_label} shown in the image? Please answer only with yes or no.'
    elif prompt_id == 9:
        prompt = f'Does this image have a {target_label}? Please answer only with yes or no.'
    elif prompt_id == 10:
        prompt = f'Is a {target_label} present in the picture? Please answer only with yes or no.'
    elif prompt_id == 11:
        prompt = f'Does the image feature a {target_label}? Please answer only with yes or no.'
    elif prompt_id == 12:
        prompt = f'Is there a {target_label} in this image? Please answer only with yes or no.'
    elif prompt_id == 13:
        prompt = f'Does this image display a {target_label}? Please answer only with yes or no.'
    elif prompt_id == 14:
        prompt = f'Can a {target_label} be seen in the image? Please answer only with yes or no.'
    elif prompt_id == 15:
        prompt = f'Is a {target_label} observable in this image? Please answer only with yes or no.'
    elif prompt_id == 16:
        prompt = f'Does this image portray a {target_label}? Please answer only with yes or no.'
    elif prompt_id == 17:
        prompt = f'Is the {target_label} present in the image? Please answer only with yes or no.'
    elif prompt_id == 18:
        prompt = f'Does this photo include a {target_label}? Please answer only with yes or no.'
    elif prompt_id == 19:
        prompt = f'Is there any {target_label} in the image? Please answer only with yes or no.'
    else:
        raise ValueError(f'Prompt id unknown {prompt_id}')
    return prompt


def load_benchmark_dictionaries(prompt_id=4) :

    data_dicts = []
    num_neg = 0
    num_pos = 0
    data_info_neg = load_json(JSON_NEG)
    for dataset in data_info_neg:
        for object_name in data_info_neg[dataset]:
            prompt = get_prompt(object_name, prompt_id)
            for fn_image in data_info_neg[dataset][object_name]:
                data_dicts.append({
                    'image_path': os.path.join(IMG_DIR_NEG, dataset, object_name, fn_image),
                    'prompt': prompt,
                    'object_name': object_name,
                    'dataset': dataset,
                    'gt':'no',
                    'img_id': fn_image,
                })
                num_neg += 1

    data_info_pos = load_json(JSON_POS)
    for dataset in data_info_pos:
        for object_name in data_info_pos[dataset]:
            prompt = get_prompt(object_name, prompt_id)
            for fn_image in data_info_pos[dataset][object_name]:
                data_dicts.append({
                    'image_path': os.path.join(IMG_DIR_POS, dataset, object_name, fn_image),
                    'prompt': prompt,
                    'object_name': object_name,
                    'dataset': dataset,
                    'gt': 'yes',
                    'img_id': fn_image,
                })
                num_pos += 1

    print(f"DASH-B: Loaded {num_neg} negative samples and {num_pos} positive samples.")
    return data_dicts