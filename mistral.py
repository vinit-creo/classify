from ctransformers import AutoModelForCausalLM
import json
BLUE = "\033[94m"
GREEN = "\033[92m"
RESET = "\033[0m"
model = AutoModelForCausalLM.from_pretrained(
    "TheBloke/Mistral-7B-v0.1-GGUF",
    model_file="mistral-7b-v0.1.Q4_K_M.gguf",
    model_type="mistral",
    gpu_layers=0 
)

def infer_intent(user_input):

    prompt = f"""
Task: Determine if the following user query is asking to see personal data or documents.
If the query is asking to show, see, intents to see, access, retrieve, or find personal data, documents, or information, classify it as "DOCUMENT_RETRIEVAL".
For all other queries, classify it as "CONVERSATION".

User query: "{user_input}"

Classification (only respond with exactly DOCUMENT_RETRIEVAL or CONVERSATION):
"""
    

    classification = model(
        prompt,
        max_new_tokens=10,  
        temperature=0.7 ,
        
    ).strip()
    
    if "DOCUMENT_RETRIEVAL" in classification:
        return {"intent": "DOCUMENT_RETRIEVAL"}
    else:
        return {"intent": "CONVERSATION"}

def process_user_query(user_input):

    intent = infer_intent(user_input)
    
    testOutput = intent
    
    if intent["intent"] == "CONVERSATION":
        conversation_prompt = f"User: {user_input}\nAssistant:"
        response = model(
            conversation_prompt,
            max_new_tokens=100,
            temperature=0.1,
            top_p=0.9
        )
    else:
        response = "I'll retrieve your personal data for you."
    
    return {
        "classification": testOutput
    }







def chatWithUser():
    print("Chat with me....:)")
    conversation = []
    
    while True:
        user_input = input(f"{BLUE}You: {RESET}")
        
        if user_input.lower() == 'quit':
            print("Goodbye!")
            break
        result = process_user_query(conversation)
        print(f"{GREEN} Mistral : {result} ")







if __name__ == "__main__":
    chatWithUser() 
