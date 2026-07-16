# Customer Complaint Labeling System

An intelligent complaint labeling tool powered by the DeepSeek API, automatically extracting complaint keywords from customer service chat logs, evaluating user sentiment scores, generating explanations, and producing a summary report.

[![CI](https://github.com/ryanchangggg/customer-complaint-labeling/actions/workflows/ci.yml/badge.svg)](https://github.com/ryanchangggg/customer-complaint-labeling/actions/workflows/ci.yml)

## Project Background

Customer service chat logs contain a wealth of high-value information. Traditional manual labeling is costly and inefficient. This project uses an LLM + Prompt approach to automate text labeling, covering complaint risk identification, sentiment assessment, and more.

Supports batch processing of hundreds of thousands of records within hours, with ~90% keyword recognition accuracy and ~80% sentiment interpretation accuracy.

## Features

- **Complaint Keyword Extraction** — Automatically identifies 10 complaint categories (course quality, investment advice, platform issues, etc.)
- **Structured Complaint Type** — Each record is classified into one of 10 predefined complaint types, with keyword-based fallback when the LLM returns an unrecognized type
- **Sentiment Score (0~10)** — Graded assessment from satisfied to extreme anger
- **Sentiment Explanation** — Generates a one-sentence explanation for each record
- **Summary Report** — After processing, an `output/report.txt` is generated with complaint type distribution, sentiment breakdown, and top keywords
- **Batch Processing** — Supports large-scale CSV file processing
- **Checkpoint Resumption** — Resumes from breakpoints after interruption without data loss (checkpoint stores only metadata, no raw text — 60%+ smaller on large datasets)
- **Auto Retry** — Automatic exponential backoff retry on API failures
- **Rate Control** — Configurable request rate limiting
- **Logging** — Detailed runtime logs for troubleshooting

## Project Structure

```
customer-complaint-labeling/
├── .github/workflows/
│   └── ci.yml               # GitHub Actions: lint + format-check + mypy + pytest (3.11–3.13)
├── config/
│   ├── config.yaml           # Global configuration
│   └── prompt.txt            # Prompt template
├── data/
│   └── sample_chat.csv       # Test dataset (500 records)
├── logs/                     # Runtime logs
├── output/                   # Output results + summary report
├── src/
│   ├── __init__.py
│   ├── main.py               # Main entry point
│   ├── config_loader.py      # Configuration loader
│   ├── prompt_manager.py     # Prompt management
│   ├── api_client.py         # DeepSeek API client
│   ├── classifier.py         # Keyword-based fallback classifier
│   ├── batch_processor.py    # Batch processing (checkpoint resume)
│   ├── reporter.py           # Summary report generator
│   └── utils.py              # Utility functions
├── tests/
│   ├── __init__.py
│   ├── test_config_loader.py
│   ├── test_api_client.py
│   └── test_batch_processor.py
├── .env                      # API Key (configure manually)
├── .gitignore
├── Makefile                  # run / test / lint / format / clean
├── pyproject.toml            # Project config + ruff + mypy + pytest
├── requirements.txt
└── README.md
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API Key

Edit the `.env` file and fill in your DeepSeek API Key:

```
DEEPSEEK_API_KEY=sk-your-deepseek-api-key-here
```

### 3. Prepare Data

Save customer service chat records as CSV format in the `data/` directory. The default test data is `data/sample_chat.csv`.

Input format:

| id | text |
|----|------|
| 1  | 为什么一直没人处理我的退款？ |
| 2  | 谢谢客服，很满意。 |

### 4. Run

```bash
# Use default configuration
python -m src.main

# Specify input and output files
python -m src.main --input data/custom.csv --output output/custom_results.csv

# Adjust batch size
python -m src.main --batch-size 10

# Specify log level
python -m src.main --log-level DEBUG

# Or use the Makefile
make run
```

### 5. View Results

The output file is located at `output/results.csv`, with the following format:

| id | text | keywords | sentiment_score | sentiment_reason | complaint_type |
|----|------|----------|----------------|-----------------|----------------|
| 1  | 会员等级降级了也没人通知 | 会员等级降级;未通知 | 5 | 用户反映会员等级降级未收到通知... | Membership/Renewal |
| 2  | 每日一股有一半以上当天是跌的 | 每日一股;一半以上;当天跌 | 5 | 用户指出推荐的股票超过一半下跌... | Investment Advice/Losses |

A **summary report** is also generated at `output/report.txt`:

```
Sentiment Score Distribution:
   0 (Satisfied             ): 107 ( 21.4%)
   2 (Neutral               ):  30 (  6.0%)
   5 (Slightly Dissatisfied ): 294 ( 58.8%)
   8 (Complaint             ):  69 ( 13.8%)
  10 (Extreme Anger         ):   0 (  0.0%)

Complaint Type Distribution:
  Course/Teaching Quality       : 163 ( 32.6%)
  Investment Advice/Losses      :  83 ( 16.6%)
  Platform/App Issues           :  65 ( 13.0%)
  Customer Service/Refund       :  41 (  8.2%)
  False Advertising             :  33 (  6.6%)
  Learning Effectiveness        :  23 (  4.6%)
  Membership/Renewal            :  10 (  2.0%)
  Pricing Issues                :  10 (  2.0%)
  Account/Order Issues          :   7 (  1.4%)
  Unfair Terms                  :   5 (  1.0%)
```

## Business Rules

### Complaint Types

| Category | Description |
|----------|-------------|
| Course/Teaching Quality | Shallow content, canceled classes, outdated material, overpriced |
| Investment Advice/Losses | Wrong stock picks, losses from recommendations, contradictory advice |
| Platform/App Issues | Data lag, crashes, missing features, sync problems |
| Customer Service/Refund | Refund rejected, poor service attitude, auto-renewal issues |
| Learning Effectiveness | Can't apply knowledge, poor progress, useless training |
| Membership/Renewal | Overpriced renewal, downgraded benefits, misleading offers |
| False Advertising | Exaggerated claims, fake credentials, bait-and-switch |
| Pricing Issues | Hidden fees, price inconsistencies, unauthorized charges |
| Unfair Terms | Restrictive refund policies, contract traps |
| Account/Order Issues | Account suspension, unauthorized charges, order errors |

### Sentiment Score Definitions

| Score | Meaning |
|-------|---------|
| 0 | Satisfied - User expresses thanks or satisfaction |
| 2 | Neutral - General inquiry or normal feedback |
| 5 | Slightly Dissatisfied - Mild complaint or discontent |
| 8 | Complaint - Significant complaint or anger |
| 10 | Extreme Anger - Strong complaint or threatening remarks |

## Custom Configuration

- **Modify Prompt**: Edit `config/prompt.txt` to adjust analysis rules and output format
- **Modify Business Rules**: Edit the `rules` section in `config/config.yaml`
- **Adjust API Parameters**: Edit the `api` section in `config/config.yaml`

## Development

### Makefile

The project includes a `Makefile` for common operations:

```bash
make run        # Run the labeling pipeline
make test       # Run tests with coverage
make lint       # Ruff lint check
make format     # Ruff auto-format
make clean      # Remove output, logs, cache
make install    # Install dependencies + dev tools
```

### Testing

```bash
# Run all tests
make test

# Quick run (no coverage)
python -m pytest tests/ -v

# With coverage report
python -m pytest tests/ --cov=src --cov-report=term
```

### Linting & Type Checking

```bash
make lint        # ruff check
make format      # ruff format
make typecheck   # mypy
```

## Tech Stack

- **Python 3.11+**
- **DeepSeek API** (OpenAI-compatible interface)
- **Pandas** / **PyYAML** / **python-dotenv** — data & config
- **Tenacity** (retry) / **tqdm** (progress bar) / **pytest** (testing)
- **Ruff** (linter + formatter) / **Mypy** (type checker)
- **GitHub Actions** (CI across 3.11–3.13)

## License

MIT
