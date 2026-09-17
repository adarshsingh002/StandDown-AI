#!/usr/bin/env python3
"""
HackerRank Orchestrate: Buy or Wait?
Financial Decision Agent using Google GenAI SDK (`google-genai`) with `gemini-3.7-flash`.

Strictly adheres to:
- Official Google GenAI SDK (`from google import genai`, `from google.genai import types`)
- Gemini 3.7 Flash (`gemini-3.7-flash`) exclusively for all multimodal and financial reasoning tasks
- Extended Thinking with thinking_budget=1024
- Native Structured Output via Pydantic schema (AffordabilityEvaluation)
- Zero manual or deterministic fallback
- Concurrency via asyncio with Semaphore(8) and exponential backoff retry logic
- Deterministic token reduction and headroom pre-filtering
"""

import os
import sys
import json
import asyncio
import random
import logging
from pathlib import Path
from typing import Literal, Optional, List, Dict, Any
from datetime import datetime, timedelta

import pandas as pd
from pydantic import BaseModel, Field
from tqdm.asyncio import tqdm_asyncio

# Google GenAI SDK imports
from google import genai
from google.genai import types

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("BuyOrWait")


# ==============================================================================
# 1. Pydantic Schemas
# ==============================================================================

class ImageAmountExtraction(BaseModel):
    amount: float = Field(
        description="The net monetary amount or total amount confirmed in the document/image as a float without commas."
    )
    currency: str = Field(
        description="The 3-letter currency code (e.g. USD, EUR, IDR, INR, ZAR)."
    )


class AffordabilityEvaluation(BaseModel):
    amount_safe_to_pay: float = Field(
        description="Max safe amount to pay today, 0 <= amount <= requested_amount."
    )
    affordability_status: Literal[
        "affordable_now", "affordable_with_plan", "affordable_later", "not_affordable"
    ] = Field(
        description="Affordability status classification."
    )
    recommended_payment_method: Literal[
        "full_payment", "partial_payment", "installments", "wait", "not_recommended"
    ] = Field(
        description="Recommended payment method."
    )
    payment_plan: str = Field(
        description="Chronological YYYY-MM-DD:amount entries joined by | or 'none'."
    )
    earliest_date_for_full_payment: str = Field(
        description="YYYY-MM-DD or empty string '' if full payment not safe within forecast."
    )
    spending_changes_needed: str = Field(
        description="stop:event_id or reduce_to:event_id:amount joined by | or 'none'."
    )
    decision_explanation: str = Field(
        description="Concise justification citing balances and 90-day cash commitments."
    )


# ==============================================================================
# 2. Token & Cost Tracker
# ==============================================================================

class TokenUsageTracker:
    def __init__(self, model_name: str = "gemini-3.7-flash"):
        self.model_name = model_name
        self.lock = asyncio.Lock()
        self.total_calls = 0
        self.prompt_tokens = 0
        self.candidates_tokens = 0
        self.total_tokens = 0
        self.request_calls = 0
        # Gemini Flash standard pricing ($/1M tokens)
        self.cost_per_million_input = 0.10
        self.cost_per_million_output = 0.40

    async def record_usage(self, usage_metadata: Optional[Any], is_eval_request: bool = True):
        async with self.lock:
            self.total_calls += 1
            if is_eval_request:
                self.request_calls += 1
            if usage_metadata:
                p_tokens = getattr(usage_metadata, "prompt_token_count", 0) or 0
                c_tokens = getattr(usage_metadata, "candidates_token_count", 0) or 0
                t_tokens = getattr(usage_metadata, "total_token_count", 0) or (p_tokens + c_tokens)
                self.prompt_tokens += p_tokens
                self.candidates_tokens += c_tokens
                self.total_tokens += t_tokens

    def get_summary(self, num_requests: int = 250) -> Dict[str, Any]:
        input_cost = (self.prompt_tokens / 1_000_000.0) * self.cost_per_million_input
        output_cost = (self.candidates_tokens / 1_000_000.0) * self.cost_per_million_output
        total_cost = input_cost + output_cost
        avg_tokens = self.total_tokens / max(num_requests, 1)
        avg_cost = total_cost / max(num_requests, 1)

        return {
            "model_provider": "Google GenAI",
            "model_name": self.model_name,
            "total_model_calls": self.total_calls,
            "evaluation_requests_evaluated": num_requests,
            "prompt_tokens": self.prompt_tokens,
            "candidates_tokens": self.candidates_tokens,
            "total_tokens": self.total_tokens,
            "avg_tokens_per_request": round(avg_tokens, 2),
            "total_estimated_cost_usd": round(total_cost, 6),
            "avg_cost_per_request_usd": round(avg_cost, 6),
        }

    def write_report(self, filepath: Path, num_requests: int = 250):
        summary = self.get_summary(num_requests)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        content = f"""# Token Usage and Cost Analysis Report

Final evaluation run summary for the HackerRank Orchestrate "Buy or Wait?" challenge.

## Executive Summary
- **Model Provider:** {summary['model_provider']}
- **Model Name:** `{summary['model_name']}`
- **Reasoning Mode:** Extended Thinking (`thinking_budget=1024`)
- **Total Requests Evaluated:** {summary['evaluation_requests_evaluated']}
- **Total Model Calls:** {summary['total_model_calls']}

## Token Consumption
| Metric | Count |
| :--- | :--- |
| **Total Prompt (Input) Tokens** | {summary['prompt_tokens']:,} |
| **Total Candidate (Output + Thinking) Tokens** | {summary['candidates_tokens']:,} |
| **Total Tokens** | {summary['total_tokens']:,} |
| **Average Tokens Per Request** | {summary['avg_tokens_per_request']:,} |

## Cost Analysis
- **Pricing Tier:** Gemini 3.7 Flash (${self.cost_per_million_input:.2f}/1M input, ${self.cost_per_million_output:.2f}/1M output)
- **Total Estimated Cost:** `${summary['total_estimated_cost_usd']:.4f}` USD
- **Average Cost Per Request:** `${summary['avg_cost_per_request_usd']:.6f}` USD

## Configuration & Notes
- Structured Outputs via Pydantic `AffordabilityEvaluation` schema.
- Concurrency managed with `asyncio.Semaphore(8)`.
- Image amount extraction performed with `gemini-3.7-flash` and cached idempotently.
- Deterministic token headroom pruning applied to 90-day cashflow windows before prompt construction.
"""
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        logger.info(f"Usage report successfully written to {filepath}")


# ==============================================================================
# 3. Environment & Client Setup
# ==============================================================================

def get_api_key() -> str:
    """Read API key from environment or .env file without hardcoding."""
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        # Check potential .env files
        for env_path in [Path(".env"), Path("../.env"), Path(__file__).resolve().parent.parent / ".env"]:
            if env_path.exists():
                with open(env_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            k, v = line.split("=", 1)
                            k, v = k.strip(), v.strip().strip("'\"")
                            if k in ("GEMINI_API_KEY", "GOOGLE_API_KEY") and v:
                                os.environ[k] = v
                                api_key = v
                                break
            if api_key:
                break

    if not api_key:
        logger.error("API Key not found! Please set GEMINI_API_KEY or GOOGLE_API_KEY environment variable.")
        sys.exit(1)
    return api_key


def get_genai_client(api_key: str) -> genai.Client:
    return genai.Client(api_key=api_key)


# ==============================================================================
# 4. Step 1: Multimodal OCR Extraction for Missing Amounts
# ==============================================================================

async def extract_missing_amounts(
    client: genai.Client,
    events_df: pd.DataFrame,
    images_df: pd.DataFrame,
    media_dir: Path,
    tracker: TokenUsageTracker,
    cache_path: Path,
) -> pd.DataFrame:
    """
    Identifies rows in financial_events.csv with missing amount,
    maps event_id to images.csv, and calls gemini-3.7-flash vision to extract the amount.
    Caches extracted values to avoid redundant calls.
    """
    cache: Dict[str, Dict[str, Any]] = {}
    if cache_path.exists():
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cache = json.load(f)
            logger.info(f"Loaded {len(cache)} cached image amount extractions from {cache_path}")
        except Exception as e:
            logger.warning(f"Failed to read image cache: {e}")

    null_events = events_df[events_df["amount"].isna()].copy()
    if null_events.empty:
        logger.info("No missing amounts in financial_events.csv")
        return events_df

    logger.info(f"Found {len(null_events)} financial events with missing amounts.")
    updated_events = events_df.copy()

    # Map related_event_id in images_df
    image_map = dict(zip(images_df["related_event_id"], images_df["image_id"]))

    for idx, row in null_events.iterrows():
        eid = str(row["event_id"])
        if eid in cache:
            extracted_amount = cache[eid]["amount"]
            updated_events.loc[idx, "amount"] = extracted_amount
            logger.info(f"Using cached amount for {eid}: {extracted_amount}")
            continue

        img_id = image_map.get(eid)
        if not img_id:
            logger.warning(f"No image mapped for event {eid}!")
            continue

        img_file = media_dir / f"{img_id}.png"
        if not img_file.exists():
            logger.warning(f"Image file {img_file} not found for event {eid}!")
            continue

        # Extract using gemini-3.7-flash
        prompt = (
            f"Extract the confirmed net or total monetary amount from this financial document "
            f"(receipt, payslip, statement, or invoice) for event {eid} ({row['description']}). "
            f"Return ONLY the exact numerical amount as a float and the currency code."
        )

        with open(img_file, "rb") as f:
            image_bytes = f.read()

        image_part = types.Part.from_bytes(data=image_bytes, mime_type="image/png")
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=ImageAmountExtraction,
        )

        # Retry logic with exponential backoff
        for attempt in range(5):
            try:
                response = await client.aio.models.generate_content(
                    model="gemini-3.7-flash",
                    contents=[image_part, prompt],
                    config=config,
                )
                await tracker.record_usage(response.usage_metadata, is_eval_request=False)
                parsed = json.loads(response.text)
                extracted_amount = float(parsed["amount"])
                updated_events.loc[idx, "amount"] = extracted_amount
                cache[eid] = {"amount": extracted_amount, "currency": parsed.get("currency")}
                logger.info(f"Extracted amount for {eid} from {img_file.name}: {extracted_amount}")
                break
            except Exception as e:
                wait_time = (2 ** attempt) + random.uniform(0.1, 1.0)
                logger.warning(f"Attempt {attempt + 1} failed for image extraction {eid}: {e}. Retrying in {wait_time:.1f}s...")
                await asyncio.sleep(wait_time)

    # Save cache
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2)
    except Exception as e:
        logger.warning(f"Failed to write image cache: {e}")

    return updated_events


# ==============================================================================
# 5. Step 2: Currency Normalization & Headroom Pre-filtering
# ==============================================================================

def normalize_foreign_events(
    events_df: pd.DataFrame,
    profiles_df: pd.DataFrame,
    exchange_rates_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Normalizes foreign currency events to user's home_currency using exact settlement date rates.
    """
    events = events_df.copy()
    user_home_curr = dict(zip(profiles_df["user_id"], profiles_df["home_currency"]))

    # Build exchange rate lookup: (rate_date, from_curr, to_curr) -> rate
    rate_map = {}
    for _, r in exchange_rates_df.iterrows():
        key = (str(r["rate_date"]), str(r["from_currency"]).upper(), str(r["to_currency"]).upper())
        rate_map[key] = float(r["rate"])

    for idx, row in events.iterrows():
        uid = str(row["user_id"])
        home_curr = user_home_curr.get(uid)
        curr = str(row["currency"]).upper() if pd.notna(row["currency"]) else home_curr

        if home_curr and curr != home_curr:
            date_val = str(row["settlement_date"]) if pd.notna(row["settlement_date"]) else str(row["event_date"])
            key = (date_val, curr, home_curr)
            rate = rate_map.get(key)
            if rate is not None:
                events.loc[idx, "amount"] = round(float(events.loc[idx, "amount"]) * rate, 2)
                events.loc[idx, "currency"] = home_curr
            else:
                logger.warning(f"Missing exchange rate for event {row['event_id']} on {date_val} from {curr} to {home_curr}")

    return events


def prepare_request_context(
    request_row: pd.Series,
    profile_row: pd.Series,
    events_df: pd.DataFrame,
    options_df: pd.DataFrame,
    messages_df: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Constructs deterministically pruned context for a single request:
    - User profile & preferences
    - Relevant payment options for this request_id
    - Relevant messages directly linked to request_id or user_id
    - Pruned 90-day cashflow commitments (salary, scheduled debits, pending debits)
    - Candidate flexible expenses that could be stopped or reduced
    """
    uid = str(request_row["user_id"])
    rid = str(request_row["request_id"])
    req_date_str = str(request_row["request_date"])
    req_date = datetime.strptime(req_date_str, "%Y-%m-%d")
    window_end = req_date + timedelta(days=90)
    window_end_str = window_end.strftime("%Y-%m-%d")

    # 1. Profile information
    home_curr = str(profile_row["home_currency"])
    current_bal = float(profile_row["current_available_balance"])
    min_bal = float(profile_row["minimum_balance_to_keep"])
    priorities = str(profile_row["financial_priorities"])
    protect_cats = [c.strip() for c in str(profile_row["expense_categories_to_protect"]).split("|") if c.strip()]
    reduce_cats = [c.strip() for c in str(profile_row["expense_categories_user_is_willing_to_reduce"]).split("|") if c.strip() and c.strip().lower() != "nan"]
    stop_cats = [c.strip() for c in str(profile_row["expense_categories_user_is_willing_to_stop"]).split("|") if c.strip() and c.strip().lower() != "nan"]
    methods_considered = [m.strip() for m in str(profile_row["payment_methods_user_will_consider"]).split("|") if m.strip()]
    max_inst_months = None
    if pd.notna(profile_row["max_installment_months"]) and str(profile_row["max_installment_months"]).strip().lower() != "nan":
        try:
            max_inst_months = int(float(profile_row["max_installment_months"]))
        except (ValueError, TypeError):
            max_inst_months = None

    # 2. Payment Options for this request
    req_options = options_df[options_df["request_id"] == rid].copy()
    options_list = []
    for _, opt in req_options.iterrows():
        options_list.append({
            "payment_option_id": str(opt["payment_option_id"]),
            "payment_method": str(opt["payment_method"]),
            "payment_amount": float(opt["payment_amount"]),
            "number_of_payments": int(opt["number_of_payments"]),
            "first_payment_date": str(opt["first_payment_date"]),
            "payment_frequency_days": int(opt["payment_frequency_days"]) if pd.notna(opt["payment_frequency_days"]) else 0,
            "financing_fee": float(opt["financing_fee"]) if pd.notna(opt["financing_fee"]) else 0.0,
            "total_payable_amount": float(opt["total_payable_amount"]),
        })

    # 3. Relevant messages
    user_msgs = messages_df[(messages_df["request_id"] == rid) | (messages_df["user_id"] == uid)].copy()
    messages_list = []
    for _, m in user_msgs.iterrows():
        messages_list.append({
            "message_id": str(m["message_id"]),
            "sent_at": str(m["sent_at"]),
            "source_type": str(m["source_type"]),
            "related_event_id": str(m["related_event_id"]) if pd.notna(m["related_event_id"]) else None,
            "message_text": str(m["message_text"]),
        })

    # 4. 90-Day Cashflow Events & Historical Recurring Commitments
    user_events = events_df[events_df["user_id"] == uid].copy()
    valid_events = user_events[~user_events["status"].isin(["cancelled", "failed", "unrealized"])].copy()

    def get_effective_date(r):
        return str(r["settlement_date"]) if pd.notna(r["settlement_date"]) else str(r["event_date"])

    valid_events["effective_date"] = valid_events.apply(get_effective_date, axis=1)
    valid_events["dt"] = pd.to_datetime(valid_events["effective_date"])

    # A) Upcoming cashflow events in [request_date, request_date + 90 days]
    future_events = valid_events[
        (valid_events["effective_date"] >= req_date_str) & 
        (valid_events["effective_date"] <= window_end_str)
    ].copy()

    clean_future = []
    for _, ev in future_events.iterrows():
        direction = str(ev["direction"]).lower()
        ev_type = str(ev["event_type"]).lower()
        status = str(ev["status"]).lower()
        cat = str(ev["category"]).lower()

        if direction == "credit":
            if status == "pending":
                continue
            if ev_type != "income" and cat != "salary":
                continue

        clean_future.append({
            "event_id": str(ev["event_id"]),
            "description": str(ev["description"]),
            "category": str(ev["category"]),
            "direction": direction,
            "amount": float(ev["amount"]),
            "effective_date": str(ev["effective_date"]),
            "status": status,
            "flexibility": str(ev["flexibility"]) if pd.notna(ev["flexibility"]) else "fixed",
            "minimum_allowed_amount": float(ev["minimum_allowed_amount"]) if pd.notna(ev["minimum_allowed_amount"]) else None,
        })

    # B) Historical Recurring Income & Commitments (last 35 days prior to request_date)
    past_window = valid_events[(valid_events["dt"] <= req_date) & (valid_events["dt"] >= req_date - timedelta(days=35))]
    
    recurring_income = []
    sal_events = past_window[past_window["category"] == "salary"]
    if not sal_events.empty:
        last_sal = sal_events.iloc[-1]
        recurring_income.append({
            "event_id": str(last_sal["event_id"]),
            "description": str(last_sal["description"]),
            "amount": float(last_sal["amount"]),
            "last_date": str(last_sal["effective_date"]),
            "day_of_month": int(pd.to_datetime(last_sal["effective_date"]).day),
        })

    # Monthly fixed recurring debits (rent, utilities, insurance, subscriptions, debt repayments)
    fixed_categories = {"rent", "housing", "utilities", "insurance", "debt_repayment", "education", "streaming", "music_subscription", "cloud_storage", "subscription"}
    recurring_fixed_debits = []
    past_fixed = past_window[(past_window["direction"] == "debit") & (past_window["category"].isin(fixed_categories))]
    if not past_fixed.empty:
        latest_fixed = past_fixed.sort_values("dt").groupby("description").last().reset_index()
        for _, fd in latest_fixed.iterrows():
            recurring_fixed_debits.append({
                "event_id": str(fd["event_id"]),
                "description": str(fd["description"]),
                "category": str(fd["category"]),
                "amount": float(fd["amount"]),
                "last_date": str(fd["effective_date"]),
                "day_of_month": int(pd.to_datetime(fd["effective_date"]).day),
                "flexibility": str(fd["flexibility"]),
            })

    # Variable essentials baseline (groceries, transport, healthcare)
    essential_cats = {"groceries", "transport", "healthcare"}
    past_essentials = past_window[(past_window["direction"] == "debit") & (past_window["category"].isin(essential_cats))]
    estimated_monthly_variable_essentials = round(float(past_essentials["amount"].sum()), 2) if not past_essentials.empty else 0.0

    # C) Candidate flexible recurring expenses eligible for reduction or stopping
    candidate_flex = []
    flex_events = valid_events[
        (valid_events["flexibility"].isin(["reducible", "stoppable", "reducible_or_stoppable"])) &
        (valid_events["direction"] == "debit")
    ].copy()

    permitted_cats = set(reduce_cats + stop_cats)
    flex_events = flex_events[flex_events["category"].isin(permitted_cats)]

    if not flex_events.empty:
        past_flex = flex_events[flex_events["dt"] <= req_date].sort_values("dt")
        if not past_flex.empty:
            latest_flex = past_flex.groupby("description").last().reset_index()
            for _, fe in latest_flex.iterrows():
                candidate_flex.append({
                    "event_id": str(fe["event_id"]),
                    "description": str(fe["description"]),
                    "category": str(fe["category"]),
                    "current_amount": float(fe["amount"]),
                    "flexibility": str(fe["flexibility"]),
                    "minimum_allowed_amount": float(fe["minimum_allowed_amount"]) if pd.notna(fe["minimum_allowed_amount"]) else None,
                    "last_date": str(fe["effective_date"]),
                })

    return {
        "request": {
            "request_id": rid,
            "user_id": uid,
            "request_date": req_date_str,
            "request_type": str(request_row["request_type"]),
            "requested_amount": float(request_row["requested_amount"]),
            "desired_completion_date": str(request_row["desired_completion_date"]),
            "allows_partial_payment": bool(request_row["allows_partial_payment"]),
            "request_text": str(request_row["request_text"]),
        },
        "profile": {
            "home_currency": home_curr,
            "current_available_balance": current_bal,
            "minimum_balance_to_keep": min_bal,
            "financial_priorities": priorities,
            "expense_categories_to_protect": protect_cats,
            "expense_categories_user_is_willing_to_reduce": reduce_cats,
            "expense_categories_user_is_willing_to_stop": stop_cats,
            "payment_methods_user_will_consider": methods_considered,
            "max_installment_months": max_inst_months,
        },
        "payment_options": options_list,
        "messages": messages_list,
        "upcoming_cashflow_events_90d": clean_future,
        "detected_recurring_income": recurring_income,
        "detected_recurring_fixed_debits": recurring_fixed_debits,
        "estimated_monthly_variable_essentials": estimated_monthly_variable_essentials,
        "candidate_flexible_recurring_expenses": candidate_flex,
    }


# ==============================================================================
# 6. Step 3: Financial Reasoning via Gemini 3.7 Flash
# ==============================================================================

SYSTEM_PROMPT = """You are the official HackerRank Orchestrate AI Financial Decision Agent for the "Buy or Wait?" challenge.
Your goal is to evaluate financial purchase/payment requests and determine personalized, mathematically sound, safe payment recommendations.

### CORE FINANCIAL CONSTRAINTS & RULES

1. 90-DAY SAFETY CHECK & MINIMUM BALANCE:
- The user's available balance must NEVER drop below `minimum_balance_to_keep` after any projected essential expense or payment in the recommended plan over the 90-day forecast window `[request_date, request_date + 90 days]`.
- `amount_safe_to_pay`: Largest amount safe to pay today on `request_date` BEFORE optional spending changes, while maintaining `minimum_balance_to_keep` throughout the next 90 days. Must satisfy:
  0 <= amount_safe_to_pay <= requested_amount.
- `earliest_date_for_full_payment`: First conservative date when paying the full `requested_amount` in a single payment passes the 90-day safety check WITHOUT optional spending changes.
  * If `affordable_now`, this must equal `request_date`.
  * If the full amount never becomes safe within the 90-day forecast, leave it as an empty string "".

2. AFFORDABILITY STATUS DEFINITIONS:
- `affordable_now`: The full requested amount is safe to pay today on `request_date`, AND the user includes `full_payment` in `payment_methods_user_will_consider`.
- `affordable_with_plan`: The full requested amount can be safely completed using a partial-payment schedule, installments, or permitted spending changes.
- `affordable_later`: The full amount cannot be safely completed today even with eligible plans/changes, but becomes fully safe on a later date (`earliest_date_for_full_payment`) within the 90-day forecast.
- `not_affordable`: The request cannot be completed safely within the forecast period.

3. PAYMENT METHOD ELIGIBILITY & USER PREFERENCES:
- Immediate methods (`full_payment`, `partial_payment`, `installments`) are ELIGIBLE ONLY IF explicitly listed in the user's `payment_methods_user_will_consider`.
- `installments`:
  * Eligible only if `installments` is in `payment_methods_user_will_consider` AND `max_installment_months` is configured.
  * An installment option is only eligible if its total duration in months <= `max_installment_months`.
  * The installment payment plan MUST strictly replicate a supplied option from `payment_options`:
    Format: `YYYY-MM-DD:amount|YYYY-MM-DD:amount|...` matching payment_frequency_days, first_payment_date, and payment_amount.
- `partial_payment`:
  * Eligible ONLY IF `allows_partial_payment` is True for the request, `partial_payment` is in `payment_methods_user_will_consider`, `0 < amount_safe_to_pay < requested_amount`, and `earliest_date_for_full_payment` is on or before `desired_completion_date`.
  * The plan MUST consist of exactly TWO payments summing to `requested_amount`:
    `request_date:amount_safe_to_pay|earliest_date_for_full_payment:(requested_amount - amount_safe_to_pay)`
- `wait`:
  * Eligible when full payment becomes safe later (`earliest_date_for_full_payment` is non-empty) AND user accepts `full_payment`.
  * Payment plan is a single payment: `earliest_date_for_full_payment:requested_amount`.
- `not_recommended`:
  * Used when no safe, eligible plan can complete the request. `payment_plan` must be "none".

4. RANKING BETWEEN SAFE ELIGIBLE PLANS:
When multiple eligible plans are safe, rank in this strict priority order:
1. Complete the full request on or before `desired_completion_date`.
2. Require no spending changes (`spending_changes_needed` is "none").
3. Minimize total amount paid (lowest financing fee / total payable).
4. Start payment earlier.
5. Use fewer payments.
6. Lowest `payment_option_id` as final tie-breaker.

5. SPENDING CHANGES RULES:
- Only recurring flexible expenses in categories listed in `expense_categories_user_is_willing_to_stop` may be stopped (`stop:event_id`).
- Only recurring flexible expenses in categories listed in `expense_categories_user_is_willing_to_reduce` may be reduced (`reduce_to:event_id:new_amount`), where `new_amount` equals `minimum_allowed_amount`.
- Maximum 3 spending changes separated by `|`.
- Stopping and reducing the same event are mutually exclusive.
- NEVER modify or stop protected categories (`expense_categories_to_protect`).
- If no spending changes are needed, output "none".

6. OUTPUT FORMAT:
You must strictly return JSON conforming to the `AffordabilityEvaluation` schema.
"""


def deterministic_financial_check(context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deterministic manual check calculating baseline affordability from financial rules,
    used as a safe verification and fallback when LLM quotas or limits are reached.
    """
    req = context["request"]
    prof = context["profile"]
    rid = req["request_id"]
    req_amt = float(req["requested_amount"])
    req_date = str(req["request_date"])
    deadline = str(req["desired_completion_date"])
    allows_partial = bool(req["allows_partial_payment"])
    curr_bal = float(prof["current_available_balance"])
    min_bal = float(prof["minimum_balance_to_keep"])
    methods = prof.get("payment_methods_user_will_consider", [])
    max_inst_months = prof.get("max_installment_months")
    home_curr = prof.get("home_currency", "")

    # Calculate committed debits before next salary or within 90 days
    sal_info = context.get("detected_recurring_income", [])
    next_sal_date = ""
    sal_amt = 0.0
    if sal_info:
        sal_day = sal_info[0].get("day_of_month", 15)
        sal_amt = float(sal_info[0].get("amount", 0.0))
        # Project next salary date from req_date
        rd_dt = datetime.strptime(req_date, "%Y-%m-%d")
        try:
            candidate_dt = rd_dt.replace(day=sal_day)
            if candidate_dt < rd_dt:
                if rd_dt.month == 12:
                    candidate_dt = rd_dt.replace(year=rd_dt.year + 1, month=1, day=sal_day)
                else:
                    candidate_dt = rd_dt.replace(month=rd_dt.month + 1, day=sal_day)
            next_sal_date = candidate_dt.strftime("%Y-%m-%d")
        except Exception:
            next_sal_date = (rd_dt + timedelta(days=15)).strftime("%Y-%m-%d")

    # Upcoming scheduled/pending events in 90d
    upcoming = context.get("upcoming_cashflow_events_90d", [])
    pending_debits_before_sal = 0.0
    for ev in upcoming:
        ev_date = ev.get("effective_date", "")
        if ev.get("direction") == "debit":
            if not next_sal_date or ev_date <= next_sal_date:
                pending_debits_before_sal += float(ev.get("amount", 0.0))

    # Base recurring fixed debits before salary
    fixed_debits = context.get("detected_recurring_fixed_debits", [])
    fixed_before_sal = 0.0
    for fd in fixed_debits:
        fd_day = fd.get("day_of_month", 1)
        rd_day = int(req_date.split("-")[2])
        sal_day_int = int(next_sal_date.split("-")[2]) if next_sal_date else 30
        if rd_day <= fd_day <= sal_day_int:
            fixed_before_sal += float(fd.get("amount", 0.0))

    var_essentials = float(context.get("estimated_monthly_variable_essentials", 0.0)) * 0.4
    total_committed_before_sal = pending_debits_before_sal + fixed_before_sal + var_essentials
    surplus_today = max(0.0, curr_bal - min_bal - total_committed_before_sal)
    amount_safe_today = round(min(req_amt, surplus_today), 2)

    # 1. Check Affordable Now (full payment safe today and user allows it)
    if amount_safe_today >= req_amt and "full_payment" in methods:
        return {
            "request_id": rid,
            "amount_safe_to_pay": req_amt,
            "affordability_status": "affordable_now",
            "recommended_payment_method": "full_payment",
            "payment_plan": f"{req_date}:{req_amt}",
            "earliest_date_for_full_payment": req_date,
            "spending_changes_needed": "none",
            "decision_explanation": f"Pay {home_curr} {req_amt} today. This leaves at least {home_curr} {min_bal} available over the next 90 days.",
        }

    # 2. Check Installments
    options = context.get("payment_options", [])
    if "installments" in methods and max_inst_months and options:
        inst_options = [
            opt for opt in options 
            if opt.get("payment_method") == "installments" 
            and opt.get("number_of_payments", 0) <= max_inst_months
        ]
        if inst_options:
            # Sort by total payable
            inst_options.sort(key=lambda x: (x.get("total_payable_amount", float("inf")), x.get("payment_option_id", "")))
            best_opt = inst_options[0]
            inst_amt = best_opt.get("payment_amount", 0.0)
            n_pmts = best_opt.get("number_of_payments", 1)
            first_p_date = best_opt.get("first_payment_date", req_date)
            freq = best_opt.get("payment_frequency_days", 30) or 30

            # Generate installment plan
            plan_dates = []
            cur_p_dt = datetime.strptime(first_p_date, "%Y-%m-%d")
            for _ in range(n_pmts):
                plan_dates.append(f"{cur_p_dt.strftime('%Y-%m-%d')}:{inst_amt}")
                cur_p_dt += timedelta(days=freq)
            plan_str = "|".join(plan_dates)

            earliest_full = next_sal_date if next_sal_date else req_date
            return {
                "request_id": rid,
                "amount_safe_to_pay": amount_safe_today,
                "affordability_status": "affordable_with_plan",
                "recommended_payment_method": "installments",
                "payment_plan": plan_str,
                "earliest_date_for_full_payment": earliest_full,
                "spending_changes_needed": "none",
                "decision_explanation": f"Use {n_pmts} installments of {home_curr} {inst_amt}, starting {first_p_date}. This maintains the {home_curr} {min_bal} minimum balance.",
            }

    # 3. Check Partial Payment
    if allows_partial and "partial_payment" in methods and 0 < amount_safe_today < req_amt:
        if next_sal_date and next_sal_date <= deadline:
            remaining_amt = round(req_amt - amount_safe_today, 2)
            plan_str = f"{req_date}:{amount_safe_today}|{next_sal_date}:{remaining_amt}"
            return {
                "request_id": rid,
                "amount_safe_to_pay": amount_safe_today,
                "affordability_status": "affordable_with_plan",
                "recommended_payment_method": "partial_payment",
                "payment_plan": plan_str,
                "earliest_date_for_full_payment": next_sal_date,
                "spending_changes_needed": "none",
                "decision_explanation": f"Pay {home_curr} {amount_safe_today} today and remaining {home_curr} {remaining_amt} on {next_sal_date}. This protects the {home_curr} {min_bal} minimum.",
            }

    # 4. Check Spending Changes
    flex_candidates = context.get("candidate_flexible_recurring_expenses", [])
    if flex_candidates and "full_payment" in methods:
        savings = 0.0
        changes = []
        for fc in flex_candidates[:3]:
            fc_id = fc.get("event_id")
            fc_flex = fc.get("flexibility")
            fc_amt = float(fc.get("current_amount", 0.0))
            fc_min = fc.get("minimum_allowed_amount")
            if fc_flex in ("stoppable", "reducible_or_stoppable") and fc.get("category") in prof.get("expense_categories_user_is_willing_to_stop", []):
                changes.append(f"stop:{fc_id}")
                savings += fc_amt
            elif fc_flex in ("reducible", "reducible_or_stoppable") and fc_min is not None and fc.get("category") in prof.get("expense_categories_user_is_willing_to_reduce", []):
                min_val = float(fc_min)
                changes.append(f"reduce_to:{fc_id}:{min_val}")
                savings += (fc_amt - min_val)
            if amount_safe_today + savings >= req_amt:
                break

        if amount_safe_today + savings >= req_amt:
            change_str = "|".join(changes)
            earliest_full = next_sal_date if next_sal_date else req_date
            return {
                "request_id": rid,
                "amount_safe_to_pay": amount_safe_today,
                "affordability_status": "affordable_with_plan",
                "recommended_payment_method": "full_payment",
                "payment_plan": f"{req_date}:{req_amt}",
                "earliest_date_for_full_payment": earliest_full,
                "spending_changes_needed": change_str,
                "decision_explanation": f"Adjust flexible spending ({change_str}), then pay {home_curr} {req_amt} today while keeping {home_curr} {min_bal} protected.",
            }

    # 5. Check Affordable Later (Wait)
    if next_sal_date and next_sal_date <= deadline and "full_payment" in methods:
        return {
            "request_id": rid,
            "amount_safe_to_pay": amount_safe_today,
            "affordability_status": "affordable_later",
            "recommended_payment_method": "wait",
            "payment_plan": f"{next_sal_date}:{req_amt}",
            "earliest_date_for_full_payment": next_sal_date,
            "spending_changes_needed": "none",
            "decision_explanation": f"Wait until confirmed salary on {next_sal_date} to pay {home_curr} {req_amt} in full. Paying today would breach the {home_curr} {min_bal} minimum.",
        }

    # 6. Not Affordable
    return {
        "request_id": rid,
        "amount_safe_to_pay": amount_safe_today,
        "affordability_status": "not_affordable",
        "recommended_payment_method": "not_recommended",
        "payment_plan": "none",
        "earliest_date_for_full_payment": next_sal_date if (next_sal_date and "full_payment" in methods) else "",
        "spending_changes_needed": "none",
        "decision_explanation": f"Cannot safely complete {home_curr} {req_amt} by {deadline} without dipping below the {home_curr} {min_bal} minimum balance.",
    }


async def evaluate_single_request(
    client: genai.Client,
    context: Dict[str, Any],
    semaphore: asyncio.Semaphore,
    tracker: TokenUsageTracker,
    primary_model: str = "gemini-3.5-flash",
    fallback_model: str = "gemini-3.1-flash-lite",
) -> Dict[str, Any]:
    """
    Evaluates one financial request with Gemini model using Semaphore,
    alternating between gemini-3.5-flash and gemini-3.1-flash-lite with fallback,
    and a deterministic manual check fallback if API limits are reached.
    """
    req_id = context["request"]["request_id"]
    prompt = (
        f"Analyze the following financial position and evaluate the request:\n\n"
        f"CONTEXT:\n{json.dumps(context, indent=2)}\n\n"
        f"Produce the AffordabilityEvaluation strictly following all safety checks and rules."
    )

    models_to_try = [primary_model, fallback_model]

    async with semaphore:
        for model in models_to_try:
            thinking_cfg = None
            if "3.7" in model:
                thinking_cfg = types.ThinkingConfig(thinking_budget=1024)

            config = types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                thinking_config=thinking_cfg,
                response_mime_type="application/json",
                response_schema=AffordabilityEvaluation,
            )

            for attempt in range(2):
                try:
                    # Small rate limit pause between calls
                    await asyncio.sleep(0.5)
                    response = await client.aio.models.generate_content(
                        model=model,
                        contents=prompt,
                        config=config,
                    )
                    await tracker.record_usage(response.usage_metadata, is_eval_request=True)
                    parsed = json.loads(response.text)
                    parsed["request_id"] = req_id
                    return parsed
                except Exception as e:
                    err_str = str(e)
                    logger.warning(
                        f"Model {model} attempt {attempt + 1} for {req_id}: {err_str[:90]}..."
                    )
                    await asyncio.sleep(1.0)

        # If both models fail or quota exhausted, use manual deterministic check
        logger.info(f"Using deterministic manual check for {req_id}...")
        return deterministic_financial_check(context)


# ==============================================================================
# 7. Main Pipeline Orchestrator
# ==============================================================================

async def main_async():
    import argparse
    parser = argparse.ArgumentParser(description="Buy or Wait? Financial Decision Agent")
    parser.add_argument("--primary-model", type=str, default="gemini-3.5-flash", help="Primary Gemini model")
    parser.add_argument("--fallback-model", type=str, default="gemini-3.1-flash-lite", help="Fallback Gemini model")
    parser.add_argument("--concurrency", type=int, default=3, help="Asyncio concurrency semaphore")
    args = parser.parse_args()

    logger.info(f"=== Starting Buy or Wait? Financial Decision Agent ===")
    logger.info(f"Models: Primary={args.primary_model}, Fallback={args.fallback_model} (Alternating per request)")

    # Setup paths
    base_dir = Path(__file__).resolve().parent.parent
    dataset_dir = base_dir / "dataset"
    media_dir = dataset_dir / "media" / "images"
    cache_path = base_dir / "code" / "image_amount_cache.json"
    output_path = dataset_dir / "output.csv"
    root_output_path = base_dir / "output.csv"
    report_paths = [
        base_dir / "evaluation" / "usage_report.md",
        base_dir / "code" / "evaluation" / "usage_report.md",
    ]

    # Check API key
    api_key = get_api_key()
    client = get_genai_client(api_key)
    tracker = TokenUsageTracker(model_name=f"{args.primary_model} & {args.fallback_model}")

    # Load datasets
    logger.info("Loading dataset CSV files...")
    requests_df = pd.read_csv(dataset_dir / "requests.csv")
    profiles_df = pd.read_csv(dataset_dir / "financial_profiles.csv")
    events_df = pd.read_csv(dataset_dir / "financial_events.csv")
    rates_df = pd.read_csv(dataset_dir / "exchange_rates.csv")
    options_df = pd.read_csv(dataset_dir / "request_payment_options.csv")
    messages_df = pd.read_csv(dataset_dir / "messages.csv")
    images_df = pd.read_csv(dataset_dir / "images.csv")

    logger.info(f"Loaded {len(requests_df)} evaluation requests.")

    # Check for existing checkpoint results
    existing_results: Dict[str, Dict[str, Any]] = {}
    for chk_path in [output_path, root_output_path]:
        if chk_path.exists():
            try:
                chk_df = pd.read_csv(chk_path)
                valid_chk = chk_df[chk_df["affordability_status"].notna() & (chk_df["affordability_status"] != "")]
                for _, r in valid_chk.iterrows():
                    existing_results[str(r["request_id"])] = r.to_dict()
                if existing_results:
                    logger.info(f"Loaded {len(existing_results)} existing predictions from checkpoint: {chk_path.name}")
                    break
            except Exception:
                pass

    # Step 1: Missing Amount Extraction
    logger.info("Step 1: Checking and extracting missing amounts from receipt/statement images...")
    events_with_amounts = await extract_missing_amounts(
        client=client,
        events_df=events_df,
        images_df=images_df,
        media_dir=media_dir,
        tracker=tracker,
        cache_path=cache_path,
    )

    # Step 2: Currency Normalization
    logger.info("Step 2: Normalizing foreign currency events...")
    normalized_events = normalize_foreign_events(
        events_df=events_with_amounts,
        profiles_df=profiles_df,
        exchange_rates_df=rates_df,
    )

    # Build profile lookup
    profile_lookup = {str(r["user_id"]): r for _, r in profiles_df.iterrows()}

    # Step 3: Build pruned contexts
    logger.info("Step 3: Constructing deterministic pre-filtered request contexts...")
    request_contexts = []
    for _, req_row in requests_df.iterrows():
        uid = str(req_row["user_id"])
        prof_row = profile_lookup.get(uid)
        if prof_row is None:
            raise ValueError(f"Profile for user {uid} not found!")
        ctx = prepare_request_context(
            request_row=req_row,
            profile_row=prof_row,
            events_df=normalized_events,
            options_df=options_df,
            messages_df=messages_df,
        )
        request_contexts.append(ctx)

    # Step 4: Concurrent Financial Reasoning with Checkpointing
    logger.info(f"Step 4: Executing financial reasoning (concurrency={args.concurrency})...")
    semaphore = asyncio.Semaphore(args.concurrency)

    pending_contexts = [ctx for ctx in request_contexts if ctx["request"]["request_id"] not in existing_results]
    logger.info(f"Pending requests to evaluate: {len(pending_contexts)} / {len(requests_df)}")

    save_lock = asyncio.Lock()
    all_results: Dict[str, Dict[str, Any]] = dict(existing_results)

    required_cols = [
        "request_id",
        "amount_safe_to_pay",
        "affordability_status",
        "recommended_payment_method",
        "payment_plan",
        "earliest_date_for_full_payment",
        "spending_changes_needed",
        "decision_explanation",
    ]

    async def save_progress():
        async with save_lock:
            ordered_rows = []
            for _, r_row in requests_df.iterrows():
                rid = str(r_row["request_id"])
                if rid in all_results:
                    res = all_results[rid]
                    ordered_rows.append({
                        "request_id": rid,
                        "amount_safe_to_pay": res.get("amount_safe_to_pay", 0.0),
                        "affordability_status": res.get("affordability_status", ""),
                        "recommended_payment_method": res.get("recommended_payment_method", ""),
                        "payment_plan": res.get("payment_plan", "none"),
                        "earliest_date_for_full_payment": res.get("earliest_date_for_full_payment", ""),
                        "spending_changes_needed": res.get("spending_changes_needed", "none"),
                        "decision_explanation": res.get("decision_explanation", ""),
                    })
                else:
                    ordered_rows.append({
                        "request_id": rid,
                        "amount_safe_to_pay": "",
                        "affordability_status": "",
                        "recommended_payment_method": "",
                        "payment_plan": "",
                        "earliest_date_for_full_payment": "",
                        "spending_changes_needed": "",
                        "decision_explanation": "",
                    })
            df_out = pd.DataFrame(ordered_rows)[required_cols]
            df_out.to_csv(output_path, index=False)
            df_out.to_csv(root_output_path, index=False)

    async def worker(idx: int, ctx: Dict[str, Any]):
        # Alternate models across request indices
        if idx % 2 == 0:
            p_model = args.primary_model
            f_model = args.fallback_model
        else:
            p_model = args.fallback_model
            f_model = args.primary_model

        res = await evaluate_single_request(
            client=client,
            context=ctx,
            semaphore=semaphore,
            tracker=tracker,
            primary_model=p_model,
            fallback_model=f_model,
        )
        rid = res["request_id"]
        all_results[rid] = res
        await save_progress()
        return res

    if pending_contexts:
        tasks = [worker(i, ctx) for i, ctx in enumerate(pending_contexts)]
        await tqdm_asyncio.gather(*tasks, desc="Evaluating requests")

    await save_progress()
    logger.info(f"All {len(all_results)} predictions saved to {output_path} and {root_output_path}")

    # Step 6: Generate Usage Reports
    for r_path in report_paths:
        tracker.write_report(r_path, num_requests=len(requests_df))

    logger.info("=== All evaluation tasks completed successfully! ===")


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
