from transformers import AutoTokenizer, AutoModelForCausalLM

_model_cache = {}

def get_model_and_tokenizer(model_name: str):
    if model_name not in _model_cache:
        print(f"Loading model {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model.eval()
        _model_cache[model_name] = (tokenizer, model)
    return _model_cache[model_name]