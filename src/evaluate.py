import argparse

from eval_utils import compute_results
from data import load_benchmark_dictionaries
from vlms import get_evaluator, VLMEvaluator


class CustomEvaluator(VLMEvaluator):
    def load_vlm(self, *args, **kwargs):
        """
        Loads the model and processors/tokenizers required for inference.

        """
        # Implement loading of model and processors/tokenizers here
        raise NotImplementedError()
    
    def evaluate_dataset(self, data_dicts, *args, **kwargs):
        """
        Args:
            data_dicts: list of dictionaries, each containing the following keys:
                - image_path: path to the image
                - prompt: text query 
                
        Returns a list of model response strings for each image-query in the dictionaries.
        """
        # Implement model evaluation here
        raise NotImplementedError()


def parse_arguments():
    parser = argparse.ArgumentParser(description='Evaluate a VLM on the DASH-B benchmark.')
    parser.add_argument('--vlm_name', type=str, default='AIDC-AI/Ovis2-1B', 
                        help='Name of the vision language model to evaluate')
    parser.add_argument('--output_dir', type=str, default='output',
                        help='Directory to save results')
    parser.add_argument('--bs', type=int, default=32,
                        help='batchsize')
    return parser.parse_args()

if __name__ == '__main__':

    args = parse_arguments()
    vlm_name = args.vlm_name    
    output_dir = args.output_dir

    print(f"Saving results to {output_dir}")
    if args.vlm_name == 'custom':
        vlm_evaluator = CustomEvaluator(vlm_name)
    else:
        vlm_evaluator = get_evaluator(vlm_name)
    benchmark_data = load_benchmark_dictionaries()

    responses = vlm_evaluator.evaluate_dataset(benchmark_data, batchsize=args.bs)
    results = compute_results(output_dir, vlm_evaluator, benchmark_data, responses)

    print(vlm_evaluator.vlm_name)
    print(results)
    