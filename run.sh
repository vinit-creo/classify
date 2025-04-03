#!/bin/bash

# Exit on error
set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' 

PYTHON_SCRIPT="llama_run.py"
VENV_NAME=".tinyllama_env"

echo -e "${BLUE}Setting up environment for TinyLlama chat application...${NC}"

if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: Python 3 is not installed.${NC}"
    echo -e "Please install Python 3 before running this script."
    exit 1
fi

# Create virtual environment
echo -e "${BLUE}Creating virtual environment '${VENV_NAME}'...${NC}"
python3 -m venv ${VENV_NAME}

# Activate virtual environment
echo -e "${BLUE}Activating virtual environment...${NC}"
source ${VENV_NAME}/bin/activate

echo -e "${BLUE}Upgrading pip...${NC}"

echo -e "${BLUE}Installing required packages...${NC}"
pip install torch transformers accelerate sentencepiece

echo -e "${BLUE}Checking hardware acceleration availability...${NC}"
python3 -c "
import torch
import platform

print('Python version:', platform.python_version())
print('PyTorch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'N/A')
print('MPS available:', hasattr(torch.backends, 'mps') and torch.backends.mps.is_available())
"

echo -e "${GREEN}Setup complete!${NC}"
echo -e "${BLUE}To run your TinyLlama chat application:${NC}"
echo -e "1. Ensure your Python script (${PYTHON_SCRIPT}) is in the current directory"
echo -e "2. Activate the virtual environment with: source ${VENV_NAME}/bin/activate"
echo -e "3. Run the script with: python ${PYTHON_SCRIPT}"
echo -e ""
echo -e "${BLUE}To run everything in one go, you can use:${NC}"
echo -e "source ${VENV_NAME}/bin/activate && python ${PYTHON_SCRIPT}"