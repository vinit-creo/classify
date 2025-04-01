from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM


def run():
    print("print")
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")











if __name__=="__main__":
    run()