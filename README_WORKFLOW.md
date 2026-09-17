# Buy or Wait? AI Agent — Workflow & Execution Guidelines

This document details the complete end-to-end workflow, architectural components, and operational guidelines for running the **Buy or Wait?** financial decision agent.

---

## 1. System Overview & Architecture

The **Buy or Wait?** AI agent processes purchase requests against a user's reconstructed financial profile, multi-currency transaction history, seller payment terms, and contextual communications.

### Core Architecture Pipeline

```text
[Dataset CSVs + Images]
       │
       ▼
1. Preprocessing & Extraction
   ├── OCR Cache (code/image_amount_cache.json) -> Resolves 16 missing amounts from PNG receipts/bills
   └── FX Normalization -> Converts foreign events using exact settlement date rates
       │
       ▼
2. Context Construction & 90-Day Cashflow Forecasting
   ├── Balances & minimum balance to keep enforcement
   ├── Reserved pending debits & confirmed future salary projection
   └── Request payment options & linked messages filtering
       │
       ▼
3. Dual-Model Hybrid Inference Engine
   ├── Primary: Alternating gemini-3.5-flash & gemini-3.1-flash-lite
   ├── Controlled Rate Limiting: 0.5s pause to prevent free/standard tier throttling
   ├── Structured Output: Pydantic AffordabilityEvaluation schema validation
   └── Deterministic Manual Check Fallback:
       Guaranteed rules-based engine when API limits, timeouts, or rate limits occur
       │
       ▼
4. Output Checkpointing & Verification
   ├── Atomic write to output.csv and dataset/output.csv per request
   └── Submission Packaging (package_submission.py -> code.zip)
```

---

## 2. Prerequisites & Installation

### Environment Requirements
- Python 3.10 or higher (Python 3.11 recommended)
- Google Gemini API Key with access to `gemini-3.5-flash` and `gemini-3.1-flash-lite`

### Setup Steps

1. **Clone and enter repository:**
   ```bash
   cd StandDown-AI
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up Gemini API Key:**
   You can either export the environment variable:
   ```bash
   export GEMINI_API_KEY="your-gemini-api-key-here"
   ```
   Or create a `.env` file in the repository root:
   ```env
   GEMINI_API_KEY=your-gemini-api-key-here
   ```
   *(The system automatically detects both `GEMINI_API_KEY` and `GOOGLE_API_KEY`).*

---

## 3. Operational Guidelines: How to Run

### Step 1: Run the Agent on All 250 Requests

To execute the financial decision engine over `dataset/requests.csv`:

```bash
python code/main.py
```

* **What happens during execution:**
  - Loads profiles, events, exchange rates, and options.
  - Resolves any missing amounts using `code/image_amount_cache.json`.
  - Normalizes currencies to the user's home currency.
  - Alternates calls between `gemini-3.5-flash` and `gemini-3.1-flash-lite`.
  - Automatically activates `deterministic_financial_check` if quota errors or rate limits occur.
  - Updates progress live and flushes each row to both `output.csv` and `dataset/output.csv`.

---

### Step 2: Evaluate Accuracy, Precision & Constraints

To benchmark model predictions against ground truth (`dataset/sample_requests.csv`):

```bash
python evaluate.py
```

* **Metrics evaluated:**
  - **Affordability Status:** Accuracy, Macro Precision, Recall, and F1 Score.
  - **Payment Method:** Accuracy, Macro Precision, Recall, and F1 Score across all options (`full_payment`, `installments`, `wait`, `not_recommended`, `partial_payment`).
  - **Numerical Amount Precision:** Mean Absolute Error (MAE), exact match percentage, and within 5% tolerance.
  - **Temporal & Constraint Adherence:** Earliest safe full payment date accuracy and spending change recommendation match rate.
  - **Overall Composite Decision Score:** Aggregate score out of 100.

---

### Step 3: Package the Solution for Submission

Run the packaging script to generate the official submission archive:

```bash
python package_submission.py
```

* **Generated file:** `code.zip` (in repository root).
* **Package contents:**
  - `code/main.py`
  - `code/image_amount_cache.json`
  - `code/evaluation/evaluate.py`
  - `code/evaluation/usage_report.md`
  - `README.md` & `requirements.txt`

---

## 4. Output Schema & Guarantees

The generated `output.csv` conforms strictly to the HackerRank challenge specification:

| Field | Type / Valid Values | Description |
|---|---|---|
| `request_id` | `str` | Must match `dataset/requests.csv` (250 rows). |
| `amount_safe_to_pay` | `float` | Amount safe on `request_date` (between 0 and `requested_amount`). |
| `affordability_status` | Enum | `affordable_now`, `affordable_with_plan`, `affordable_later`, or `not_affordable`. |
| `recommended_payment_method`| Enum | `full_payment`, `partial_payment`, `installments`, `wait`, or `not_recommended`. |
| `payment_plan` | Formatted String | Chronological `YYYY-MM-DD:amount` entries separated by `\|`, or `none`. |
| `earliest_date_for_full_payment` | `YYYY-MM-DD` or empty | Earliest date one full payment is safe; matches `request_date` if `affordable_now`. |
| `spending_changes_needed` | Formatted String | `none` or up to 3 `stop:<event_id>` / `reduce_to:<event_id>:<amount>` actions. |
| `decision_explanation` | `str` | Concise, grounded reasoning adhering to financial rules. |

---

## 5. Submission Checklist & Submission Link

Before uploading to HackerRank, confirm:
- [x] All 250 requests in `dataset/requests.csv` are evaluated in `output.csv`.
- [x] No NaN or missing values exist in `output.csv`.
- [x] `code.zip` includes `code/`, `evaluation/usage_report.md`, and dependencies.
- [x] All API keys and secrets are removed from committed files and `log.txt`.

**Mandatory Submission URL:**  
https://www.hackerrank.com/contests/hackerrank-orchestrate-september26/challenges/buy-or-wait/submission
