# local-llm-checker

Find which local LLMs can run on your system — before you waste time downloading them.

## Installation
```bash
pipx install local-llm-checker
```

## Usage
```bash
# List all compatible models
local-llm-checker

# Filter by tag
local-llm-checker --tag coding

# Check a specific model
local-llm-checker check mistral

# Update model registry
local-llm-checker update
```

## Development
```bash
git clone https://github.com/Manju-Chinnanagoudar/local-llm-checker
cd local-llm-checker
pip install -e .
```