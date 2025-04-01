from transformers import pipeline
import json

question_answerer = pipeline("question-answering", model='distilbert-base-uncased-distilled-squad',)

context = r"""
For the given user query, determine whether the user request is a DOCUMENT_RETRIEVAL or CONVERSATION. Do not give Outputs other than these. please provide a JSON response with the following structure:
{
    "intent": "value",
} 
"""
result = question_answerer(question="Show me my recent Blood Sugar records", context=context)
print(f"::: {result}")
json_output = json.dumps(result, indent=4)
print(f"{json_output}")
