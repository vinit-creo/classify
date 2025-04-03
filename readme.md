# Classify

A lightweight intent classification system using TinyLlama, designed to distinguish between document retrieval requests and general conversation.

## Overview

This repository contains a Python implementation of an intent classifier using the TinyLlama-1.1B-Chat model. The classifier analyzes user inputs and categorizes them as either:

- `DOCUMENT_RETRIEVAL`: Queries related to finding or retrieving documents
- `Conversation`: General conversation or questions not related to document retrieval

## Features

- Lightweight implementation using TinyLlama-1.1B-Chat (only 1.1B parameters)
- Support for multiple hardware acceleration backends (CUDA, MPS, CPU)
- Efficient memory management for resource-constrained environments
- Simple interactive chat interface
- JSON-formatted responses for easy integration with other systems

## Requirements

- Python 3.8+
- PyTorch
- Transformers
- SentencePiece
- Accelerate (optional, for improved performance)

## Installation

Clone the repository:

```bash
git clone https://github.com/vinit-creo/classify.git
cd classify
```

Set up the environment using the provided script:

```bash
chmod +x setup.sh
./setup.sh
```

This will create a Python virtual environment and install all required dependencies.

## Usage

After setting up the environment, you can run the classifier:

```bash
source tinyllama_env/bin/activate
python tinyllama_chat.py
```

This will start an interactive chat session where you can type queries and see their classification:

```
Chat with me... :)
You: Show me my lab results from last week
Mistral: {'intent': 'DOCUMENT_RETRIEVAL'}
You: How are you doing today?
Mistral: {'intent': 'Conversation'}
```

Type `quit` to exit the chat session.

## How It Works

The system uses a prompt-based approach to classification:
1. User input is wrapped in a system prompt that provides examples of both document retrieval and conversation intents
2. TinyLlama processes the input and generates a JSON response
3. The JSON is parsed to extract the intent classification
4. The result is displayed to the user

## Advanced Configuration

The script can be modified to adjust:
- Model parameters (temperature, max tokens)
- System prompt examples
- Output format
- Hardware acceleration options

## Integration

To integrate with other systems, you can:
1. Import the `process_text` function
2. Call it with your user text
3. Process the returned JSON object

Example:
```python
from tinyllama_chat import process_text

result = process_text("Find my medical records")
if result.get("intent") == "DOCUMENT_RETRIEVAL":
    # Handle document retrieval
else:
    # Handle conversation
```

## License

[MIT License](LICENSE)

## Acknowledgements

- This project uses the [TinyLlama](https://github.com/jzhang38/TinyLlama) model
- Built with [Hugging Face Transformers](https://github.com/huggingface/transformers)