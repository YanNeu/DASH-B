import os 
import json


def compute_answer(response):
    answer = None
    if "yes" in response.lower():
        answer = "yes"
    elif "no" in response.lower():
        answer = "no"        
    return answer


def compute_results(output_dir, vlm_evaluator, benchmark_data, responses):

    answers = []
    for response, data_dict in zip(responses, benchmark_data):
        answers.append({
            'img_id': data_dict['img_id'],
            'object_name': data_dict['object_name'],
            'gt': data_dict['gt'],
            'response': response,
            'answer': compute_answer(response)
        })

    answers_json = os.path.join(output_dir, f'{vlm_evaluator.vlm_name.replace('/', '_')}_answers.json')
    with open(answers_json, 'w') as f:
        f.write(json.dumps(answers, indent=4))

    n_correct = 0
    n_correct_neg = 0
    n_correct_pos = 0
    n_total = 0
    n_total_neg = 0
    n_total_pos = 0
    for answer_dict in answers:
        if answer_dict['answer'] == answer_dict['gt']:
            n_correct += 1
            n_correct_pos += int(answer_dict['gt'] == 'yes')
            n_correct_neg += int(answer_dict['gt'] == 'no')
        n_total += 1 
        n_total_pos += int(answer_dict['gt'] == 'yes')
        n_total_neg += int(answer_dict['gt'] == 'no')
    
    results = {
        'ACC': n_correct / n_total,
        'TNR': n_correct_neg / n_total_neg,
        'TPR': n_correct_pos / n_total_pos,
    }
    results['Harmonic Mean'] = 2 * (results['TPR'] * results['TNR']) / (results['TPR'] + results['TNR'])
    
    results_json = os.path.join(output_dir, f'{vlm_evaluator.vlm_name.replace('/', '_')}_dash_b_results.json')
    with open(results_json, 'w') as f:
        f.write(json.dumps(results, indent=4))
    return results