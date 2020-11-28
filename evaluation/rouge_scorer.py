from rouge_score import rouge_scorer
from pathlib import Path
import argparse
import os


def get_rouge_scores(target_summaries_path, predicted_summaries_path, use_stem=False):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=use_stem)
    
    target_summary_files = [x for x in Path(target_summaries_path).rglob('*.summary')]
    
    rouge1 = []
    rouge2 = []
    rougel = []

    for target_summary_path in target_summary_files:
        basename = Path(target_summary_path).stem
        with open(target_summary_path, 'r') as f:
            target_summary = f.read()
        
        predicted_summary_path = f'{predicted_summaries_path}/{basename}.summary'

        if not os.path.exists(predicted_summary_path):
            continue

        with open(predicted_summary_path, 'r') as f:
            predicted_summary = f.read()
        
        output = scorer.score(target_summary, predicted_summary)

        rouge1 += [output['rouge1'].fmeasure]
        rouge2 += [output['rouge2'].fmeasure]
        rougel += [output['rougeL'].fmeasure]

    return sum(rouge1) / len(rouge1) * 100, sum(rouge2) / len(rouge2) * 100, sum(rougel) / len(rougel) * 100


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--target_path', nargs='*', help='path to target summaries')
    parser.add_argument('-p', '--predicted_path', nargs='*', help='path to predicted summaries')
    parser.add_argument('-s', '--use_stem', nargs='*', help='whether to use stemming')
    args = parser.parse_args()

    r1, r2, rl = get_rouge_scores(args.target_path[0], args.predicted_path[0], args.use_stem[0])

    print('ROUGE-1 score:', r1)
    print('ROUGE-2 score:', r2)
    print('ROUGE-L score:', rl)
