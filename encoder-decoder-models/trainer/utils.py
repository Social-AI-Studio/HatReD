import os
from evaluate import load

class Evaluator:
    def __init__(self):
        self.bleu = load("bleu")
        self.rouge = load('rouge')
        self.bertscore = load("bertscore")
        self.meteor = load('meteor')

    def compute_metrics(self, candidates, references):
        results = {}

        # bleu
        try:
            bleu_results = self.bleu.compute(predictions=candidates, references=references)
            results['bleu'] = bleu_results['bleu']
            results['bleu-1'] = bleu_results['precisions'][0]
            results['bleu-2'] = bleu_results['precisions'][1]
            results['bleu-3'] = bleu_results['precisions'][2]
            results['bleu-4'] = bleu_results['precisions'][3]
            results['brevity-penalty'] = bleu_results['brevity_penalty']
        except:
            results['bleu'] = 0.0
            results['bleu-1'] = 0.0
            results['bleu-2'] = 0.0
            results['bleu-3'] = 0.0
            results['bleu-4'] = 0.0

        # rouge
        try:
            rouge_results = self.rouge.compute(predictions=candidates, references=references)
            results = {**results, **rouge_results}
        except:
            results['rouge1'] = 0.0
            results['rouge2'] = 0.0
            results['rougeL'] = 0.0
        
        # BERTScore
        try:
            avg = lambda lst: sum(lst) / len(lst)
            bertscore_results = self.bertscore.compute(predictions=candidates, references=references, lang="en", rescale_with_baseline=True)
            results['P-BERT'] = avg(bertscore_results['precision'])
            results['R-BERT'] = avg(bertscore_results['recall'])
            results['F-BERT'] = avg(bertscore_results['f1'])
        except:
            results['P-BERT'] = 0.0
            results['R-BERT'] = 0.0
            results['F-BERT'] = 0.0


        # meteor
        try:
            meteor_results = self.meteor.compute(predictions=candidates, references=references)
            results['meteor'] = meteor_results['meteor']
        except:
            results['meteor'] = 0.0

        # harmonic mean
        harmonic_mean = 2 * (results['bleu'] * results['rougeL']) / (results['bleu'] + results['rougeL'])
        results['harmonic-mean'] = harmonic_mean

        return results

class SaveCheckpoint:
    def __init__(self, model, model_filename, logger):
        self.results = []
        self.harmonic_means = []
        self.FBERTs = []
        
        self.model = model
        self.best_harmonic_mean_filepath = os.path.join(f"{model_filename}-best-harmonic-mean.pt")
        self.best_bert_score_filepath = os.path.join(f"{model_filename}-best-bert-score.pt")

        self.logger = logger
    
    def update_checkpoint(self, results):
        self.results.append(results)
        self.harmonic_means.append(results['harmonic-mean'])
        self.FBERTs.append(results['F-BERT'])

        self._save_model()

    def _save_model(self):
        if max(self.harmonic_means) == self.harmonic_means[-1]:
            self.logger.write("\tbest model (for harmonic mean) checkpoint found... saving model...")
            self.model.save_pretrained(self.best_harmonic_mean_filepath)

        if max(self.FBERTs) == self.FBERTs[-1]:
            self.logger.write("\tbest model (for BERTScore) checkpoint found... saving model...")
            self.model.save_pretrained(self.best_bert_score_filepath)

def get_metric_report(results: dict, key: str):
    filtered_results = [
        f"{k.upper()}: {v:.4f}" 
        for k, v in results.items()
        if key in k
    ]

    return ", ".join(filtered_results)

def get_epoch_summary(results, title="Computing overall validation metrics"):
    msg = [title]
    msg.append(f"\tval {get_metric_report(results, 'bleu')}")
    msg.append(f"\tval {get_metric_report(results, 'rouge')}")
    msg.append(f"\tval {get_metric_report(results, 'BERT')}")
    msg.append(f"\tval {get_metric_report(results, 'meteor')}")
    msg.append(f"\tval {get_metric_report(results, 'harmonic')}")

    return "\n".join(msg)

def get_overall_summary(checkpoint):
    msg = "\nReporting Best Validation Results\n\n"

    mean_idx = sorted(range(len(checkpoint.harmonic_means)),
                key=lambda k: checkpoint.harmonic_means[k],
                reverse=True)[0]
    mean_results = checkpoint.results[mean_idx]

    msg += get_epoch_summary(mean_results, f"Maximum epoch (for harmonic mean): {mean_idx}")

    bert_idx = sorted(range(len(checkpoint.FBERTs)),
                key=lambda k: checkpoint.FBERTs[k],
                reverse=True)[0]
    bert_results = checkpoint.results[bert_idx]

    msg += "\n"
    msg += get_epoch_summary(bert_results, f"Maximum epoch (for BERTScore): {bert_idx}")

    return msg