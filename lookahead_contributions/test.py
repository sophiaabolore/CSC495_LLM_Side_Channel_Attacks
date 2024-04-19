from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time 
import json
import os
import pandas as pd 
from datasets import load_dataset
# Load both explicit and implicit datasets
explicit_dataset = load_dataset("Anthropic/discrim-eval", "explicit")["train"]
implicit_dataset = load_dataset("Anthropic/discrim-eval", "implicit")["train"]
torch_device = "cuda"
if int(os.environ.get("LOAD_LADE", 0)):
    import lade 
    lade.augment_all()
    lade.config_lade(LEVEL=7, WINDOW_SIZE=20, GUESS_SET_SIZE=20, DEBUG=1)

assert torch.cuda.is_available()


def measure_latency(model, tokenizer, dataset, torch_device="cuda"):
    latencies = []
    responses = []
    data_records = []
    i = 0
    for item in dataset:
        if i >= 1000:
           break
        i+= 1
        prompt = item['filled_template']
        start_time = time.time()
        model_inputs = tokenizer(prompt, return_tensors='pt').to(torch_device)
        output = model.generate(**model_inputs, max_new_tokens=256, do_sample=False)
        end_time = time.time()
        
        latency = end_time - start_time
        response_text = tokenizer.decode(output[0], skip_special_tokens=True)
        
        latencies.append(latency)
        responses.append({'prompt': prompt, 'response': response_text, 'latency': latency})
        print("response: ", response_text)
        print("latency: ", latency)
        data_records.append({
        "filled_template": item['filled_template'],
        "age": item['age'],
        "gender": item['gender'],
        "race": item['race'],
        "latency": latency,
	"fill_type":item['fill_type'],
        "response": response_text
    })
    df = pd.DataFrame(data_records)
    print(df.head())
    df.to_csv("latency_demographics.csv", index=False)
    return responses, latencies

def main():
    # Load dataset
    explicit_dataset = load_dataset("Anthropic/discrim-eval", "explicit")['train']
    
    # Setup model and tokenizer
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v0.3"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map=torch_device, use_auth_token=True)
    
    # Measure latency for explicit dataset
    explicit_responses,explicit_latencies = measure_latency(model, tokenizer, explicit_dataset, torch_device)
    implicit_responses, implicit_latencies =  measure_latency(model, tokenizer, implicit_dataset, torch_device)
    return  explicit_responses, explicit_latencies, implicit_responses, implicit_latencies


if __name__ == "__main__":
    main()
