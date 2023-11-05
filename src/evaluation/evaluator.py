import numpy as np
import evaluate
from src.utils.predictor import Predictor
from sklearn.metrics.pairwise import cosine_similarity
import fasttext


class Evaluator:
    """
    Class is responsible for evaluating models. It calculates metric
    for text translation ('meteor', 'rogue', 'bleu', e.t.c.) and
    metrics for identifying toxicity level and content preservation
    """

    def __init__(self, tokenizer, metric: str = 'meteor', sim_model_path: str = '../models/fasttext.bin'):
        """
        :param tokenizer: Which tokenizer will be used for text decoding
        :param metric: text translation metric
        Also, it initializes models for identifying toxicity level and content preservation
        """
        self.metric = evaluate.load(metric)
        self.sta_model = Predictor('s-nlp/roberta_toxicity_classifier_v1', 'transformers', 'classification')
        self.sim_model = fasttext.load_model(sim_model_path)
        self.tokenizer = tokenizer

    @staticmethod
    def _postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        return preds, labels

    def _get_cosine_similarity(self, preds: list[str], labels: list[str], embedding_size=100):
        """
        Computes cosine similarity between embeddings
        :param preds: list of the predicted sequences
        :param labels: list of the true detoxified sequences
        :param embedding_size: size of the embedding, depends on sim_model
        """
        embeddings1 = np.zeros((len(preds), embedding_size))
        for i, pred in enumerate(preds):
            for word in pred.split():
                embeddings1[i] += self.sim_model[word]
        embeddings2 = np.zeros((len(preds), embedding_size))
        for i, label in enumerate(labels):
            for word in label.split():
                embeddings2[i] += self.sim_model[word]
        cosine_similarities = []
        for vec1, vec2 in zip(embeddings1, embeddings2):
            cosine_sim = cosine_similarity([vec1], [vec2])[0][0]
            cosine_similarities.append(cosine_sim)
        return np.array(cosine_similarities).mean()

    def compute_metric(self, eval_preds):
        """
        Computes defined metrics for the prediction of the model
        """
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
        result['SIM'] = self._get_cosine_similarity(decoded_preds, decoded_labels)
        return result
