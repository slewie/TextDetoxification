from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


class Predictor:
    """
    Class is responsible for predicting using pretrained model. It supports only models from pytorch and transformers libraries.
    """

    def __init__(self, model_name: str, library: str):
        self.model_name = model_name
        self.library = library
        self._check_library()

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
                return self._predict_transformers(request)
            case 'pytorch':
                return self._predict_pytorch(request)

    def _predict_transformers(self, request):
        model = AutoModelForSeq2SeqLM.from_pretrained(f'./models/{self.model_name}-finetuned')
        tokenizer = AutoTokenizer.from_pretrained(f'./models/tokenizers/{self.model_name}-finetuned')
        model.eval()
        model.config.use_cache = False

        input_ids = tokenizer(request, return_tensors="pt").input_ids
        outputs = model.generate(input_ids, max_new_tokens=40, do_sample=True, top_k=30, top_p=0.95)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    def _predict_pytorch(self, request):
        pass
