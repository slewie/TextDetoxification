from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, RobertaForSequenceClassification
import os
import torch.nn.functional as F


class Predictor:
    """
    Class is responsible for predicting using pretrained model. It supports only models from pytorch and transformers libraries.
    """

    def __init__(self, model_name: str, library: str, task: str):
        self.model_name = model_name
        self.library = library
        self.task = task
        self._check_library()
        self.model = None
        self.tokenizer = None
        self._download_model()

    def _download_model(self):
        if self.library == 'transformers' and self.task == 'classification':
            self.model = RobertaForSequenceClassification.from_pretrained(f'{self.model_name}')
            self.tokenizer = AutoTokenizer.from_pretrained(f'{self.model_name}')
        elif self.library == 'transformers':
            if os.path.exists(f'./models/{self.model_name}-finetuned'):
                self.model = AutoModelForSeq2SeqLM.from_pretrained(f'./models/{self.model_name}-finetuned')
                self.tokenizer = AutoTokenizer.from_pretrained(f'./models/tokenizers/{self.model_name}-finetuned')
            else:
                self.model = AutoModelForSeq2SeqLM.from_pretrained(f'{self.model_name}')
                self.tokenizer = AutoTokenizer.from_pretrained(f'{self.model_name}')

    def _check_library(self):
        """
        The function checks whether the 'library' parameter is supported or not.
        """
        if self.library not in ['pytorch', 'transformers']:
            raise NameError(
                f"'{self.library}' is not supported. "
                f"You can use only 'pytorch' and 'transformers' as parameter 'library'")

    def predict(self, request):
        """
        Predict function obtains input text request and passes it to the model and return the model output
        :param request: input text request
        :return: the output of model
        """
        match self.library:
            case 'transformers':
                match self.task:
                    case 'classification':
                        return self._predict_transformers_classification(request)
                    case 'seq2seq':
                        return self._predict_transformers_seq2seq(request)
            case 'pytorch':
                return self._predict_pytorch(request)

    def _predict_transformers_seq2seq(self, request):
        self.model.eval()
        self.model.config.use_cache = False
        input_ids = self.tokenizer(request, return_tensors="pt").input_ids
        outputs = self.model.generate(input_ids, max_new_tokens=40, do_sample=True, top_k=30, top_p=0.95)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def _predict_transformers_classification(self, request):
        if isinstance(request, list):
            results = []
            for req in request:
                encoded = self.tokenizer.encode(req, return_tensors='pt')
                results.append(F.softmax(self.model(encoded).logits, dim=1)[0][1].item())
            return results
        else:
            batch = self.tokenizer.encode(request, return_tensors='pt')
            return F.softmax(self.model(batch).logits, dim=1)[0][1].item()

    def _predict_pytorch(self, request):
        pass
