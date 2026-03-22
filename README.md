# llm-checker

Find which local LLMs can run on your system — before you waste time downloading them.

## Installation
```bash
pipx install llm-checker
```

## Usage
```bash
# List all compatible models
llm-checker

# Filter by tag
llm-checker --tag coding

# Check a specific model
llm-checker check mistral

# Update model registry
llm-checker update
```

## Development
```bash
git clone https://github.com/yourname/llm-checker
cd llm-checker
pip install -e .
```