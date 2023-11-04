import numpy as np
import evaluate
from src.utils.predictor import Predictor


class Evaluator:
    """
    Class is responsible for evaluating models. It uses metrics from huggingface
    """
    def __init__(self, tokenizer, metric: str = 'meteor'):
        self.metric = evaluate.load(metric)
        self.sta_model = Predictor('s-nlp/roberta_toxicity_classifier_v1', 'transformers', 'classification')
        self.tokenizer = tokenizer

    @staticmethod
    def _postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [[label.strip()] for label in labels]

        return preds, labels

    def compute_metric(self, eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)

        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds, decoded_labels = self._postprocess_text(decoded_preds, decoded_labels)
        sta_result = self.sta_model.predict(decoded_preds)
        result = self.metric.compute(predictions=decoded_preds, references=decoded_labels)
        result['STA'] = np.array(sta_result).mean()
        return result
