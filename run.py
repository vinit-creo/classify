
# model_name = "distilgpt2"  
model_name ="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
# model_name = "tiiuae/falcon-rw-1b"
testOutputC = {"intent": "CONVERSATION"}
testOutputD = {"intent": "DOCUMENT_RETRIEVAL"}


import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import warnings
import json
import re
import gc
import copy

class MacMiniTextProcessor:
    def __init__(self, model_name=model_name):

        self.has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        
        if self.has_mps:
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
                
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True
                )
                
                if self.has_mps:
                    try:
                        self.model = self.model.to(self.device)
                    except Exception as e:
                        self.device = torch.device("cpu")
                
            except Exception as e:                
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    low_cpu_mem_usage=True,
                    load_in_4bit=True
                )
                
                self.device = torch.device("cpu")
            
        self.model_name = model_name
        print(f"Model loaded on {self.device}")
        
        gc.collect()
        if self.has_mps:
            try:
                torch.mps.empty_cache()
            except:
                pass
    
    def process_text(self, user_text):

        system_prompt = f"""
    Analyze the input and respond in JSON with 'intent' as either 'DOCUMENT_RETRIEVAL' or 'Conversation':
    
    Examples:
    1. Input: "Show me my blood test results"
       Output: {{"intent": "DOCUMENT_RETRIEVAL"}}
    
    2. Input: "What's the weather today?"
       Output: {{"intent": "Conversation"}}
    
    3. Input: "Did my MRI report arrive?"
       Output: {{"intent": "DOCUMENT_RETRIEVAL"}}'
    4. Inpout: "How does my blood work look like"
       Output:{{"intent": "DOCUMENT_RETRIEVAL"}} 
    
    5. Input: "{user_text}"
       Output: 
    """
        
        if "llama" in self.model_name.lower() or "tinyllama" in self.model_name.lower():
            full_prompt = f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_text} [/INST]"
        elif "gpt" in self.model_name.lower():
            full_prompt = f"{system_prompt}\n\nUser: {user_text}\n\nResponse:"
        else:
            full_prompt = f"System: {system_prompt}\n\nUser: {user_text}\n\nAssistant:"
        
        try:
            inputs = self.tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=512)
            
            input_tensors = {}
            for key in inputs:
                input_tensors[key] = inputs[key].to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **input_tensors,
                    max_new_tokens=256,  
                    temperature=0.1,  
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            del input_tensors
            if self.has_mps:
                try:
                    torch.mps.empty_cache()
                except:
                    pass           
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            del outputs
            gc.collect()
            if self.has_mps:
                try:
                    torch.mps.empty_cache()
                except:
                    pass
            
            response_text = None
            
            if "llama" in self.model_name.lower() or "tinyllama" in self.model_name.lower():
                if "[/INST]" in generated_text:
                    response_text = generated_text.split("[/INST]")[-1].strip()
            elif "gpt" in self.model_name.lower():
                if "Response:" in generated_text:
                    response_text = generated_text.split("Response:")[-1].strip()
            
            if response_text is None:
                if "Assistant:" in generated_text:
                    response_text = generated_text.split("Assistant:")[-1].strip()
                elif user_text in generated_text:
                    response_text = generated_text.split(user_text)[-1].strip()
                else:
                    response_text = generated_text

                print(f":::{response_text}")    
            json_match = re.search(r'({[\s\S]*})', response_text)
            
            if json_match:
                json_str = json_match.group(1)
                print(f"::: json str {json_str}")
                try:
                    parsed_json = json.loads(json_str)
                    return parsed_json
                except json.JSONDecodeError:
                    clean_json_str = re.sub(r'```json|```', '', json_str).strip()
                    try:
                        parsed_json = json.loads(clean_json_str)
                        return parsed_json
                    except:
                        return {"error": "Failed to parse JSON", "raw_text": json_str}
            else:
                return {"error": "No JSON found in output", "raw_text": response_text}
        
        except Exception as e:
            print(f"Error during processing or generation: {e}")
            return {"error": str(e)}

def main(): 
    print("Initializing Mac Mini optimized processor...")
    processor = MacMiniTextProcessor(model_name=model_name)
    
    user_query = "How does the weather look like??"
    
    response_c = processor.process_text(user_query)
    print(json.dumps(response_c, indent=2))
    
    gc.collect()
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        try:
            torch.mps.empty_cache()
        except:
            pass
    
    response_d = processor.process_text(user_query)
    print(json.dumps(response_d, indent=2))

if __name__ == "__main__":
    main()