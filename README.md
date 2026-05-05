# Multi-Domain Support Triage Agent

A Python-based support ticket triage system for HackerRank, Claude, and Visa support issues. It classifies incoming tickets, retrieves relevant support documentation, and generates structured responses using a local Ollama LLM with rule-based fallback handling.

## Features

- Scrapes support documentation for HackerRank, Claude, and Visa
- Builds a TF-IDF retrieval index over the documentation corpus
- Classifies tickets by company, product area, urgency, and request type
- Detects high-risk issues such as fraud, stolen cards, refunds, and suspicious requests
- Generates CSV output with status, response, justification, and request type
- Supports interactive terminal mode for testing single tickets
- Uses local Ollama models, so no hosted LLM API key is required

## Project Structure

```text
.
├── agent.py
├── classifier.py
├── main.py
├── retriever.py
├── scraper.py
├── requirements.txt
└── data
    ├── corpus
    │   ├── claude.json
    │   ├── hackerrank.json
    │   └── visa.json
    └── support_issues
        ├── support_tickets.csv
        ├── sample_support_tickets.csv
        └── output.csv
```

## Setup

Install Python dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Install and run Ollama, then make sure the model is available:

```bash
ollama pull llama3.1:8b
ollama list
```

## Usage

Run the full ticket-processing pipeline:

```bash
python3 main.py --run
```

If Ollama is slow, increase the timeout:

```bash
OLLAMA_TIMEOUT=180 python3 main.py --run
```

Run in interactive mode:

```bash
python3 main.py --interactive
```

Scrape support documentation:

```bash
python3 main.py --scrape
```

Build the retrieval index:

```bash
python3 main.py --build-index
```

Run the complete pipeline:

```bash
python3 main.py --all
```

## Input

The main input file is:

```text
data/support_issues/support_tickets.csv
```

Expected input columns:

```text
ticket_id,issue,subject,company
```

## Output

The generated output is saved to:

```text
data/support_issues/output.csv
```

Output columns:

```text
ticket_id,status,product_area,response,justification,request_type
```

## Example

Input ticket:

```text
My HackerRank assessment IDE keeps crashing when I run my code.
```

Example output:

```text
status: replied
product_area: IDE & Compiler
request_type: bug
```

## Environment Variables

```text
OLLAMA_BASE_URL   Optional, defaults to http://localhost:11434
OLLAMA_MODEL      Optional, defaults to llama3.1:8b
OLLAMA_TIMEOUT    Optional, defaults to 180
```

## Notes

Generated files such as logs, Python cache files, virtual environments, TF-IDF index files, and output CSV files are ignored by Git.
