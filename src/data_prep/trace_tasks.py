# src/data_prep/trace_tasks.py

"""
Build trace-derived Tenacious-Bench preference tasks from Week 10 (The-Conversion-Engine) Langfuse traces.

What this script does:
- Preserves real generated email bodies as agent_output, not only outreach_body.
- Creates good/bad pairs using pair_id so DPO conversion can work reliably.
- Converts real Week 10 system errors into rejected examples with corrected chosen outputs.
- Adds source_mode, failure codes, trace IDs, and reproducible metadata.

Default input:  data/raw/llm_traces.jsonl
Default output: data/tasks/trace_tasks.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

BOOKING_LINK = "https://cal.com/tenacious/discovery-call-15"
LOCALHOST_LINK_PATTERN = re.compile(r"http://localhost:\d+[^\s)\]}>,]*")

BANNED_PHRASES = [
    "world-class",
    "top talent",
    "a-players",
    "rockstar",
    "ninja",
    "wizard",
    "skyrocket",
    "supercharge",
    "10x",
    "i hope this email finds you well",
    "just following up",
    "circling back",
    "quick question",
    "quick chat",
    "synergize",
    "synergy",
    "leverage",
    "ecosystem",
    "game-changer",
    "disruptor",
    "paradigm shift",
    "you'll regret",
    "don't miss out",
    "per my last email",
    "our proprietary",
    "our ai-powered",
]


def stable_id(*parts: Any, length: int = 12) -> str:
    """Create a stable ID from any number of parts."""
    raw = "::".join(str(p) for p in parts if p is not None)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:length]


def safe_json_loads(value: Any) -> Any:
    """Return a dict from a JSON string, or an empty dict if invalid."""
    if value is None:
        return {}
    if isinstance(value, (dict, list)):
        return value
    if not isinstance(value, str):
        return {}
    value = value.strip()
    if not value:
        return {}
    try:
        return json.loads(value)
    except Exception:
        return {}


def read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    """Return a generator of dicts from a JSONL file."""
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"[WARN] Skipping malformed JSONL line {line_no}: {exc}")


def clean_output_text(text: Optional[str]) -> str:
    """Clean up an output text string."""
    if not text:
        return ""
    text = LOCALHOST_LINK_PATTERN.sub(BOOKING_LINK, text)
    text = re.sub(r"\s+", " ", text.replace("\r", " ").replace("\n", " ")).strip()
    return text


def extract_output_body(output: Any) -> Tuple[str, str]:
    """Return (subject, body_or_error_text) from a Langfuse output field."""
    obj = safe_json_loads(output)
    if isinstance(obj, dict):
        subject = obj.get("subject", "") or ""
        body = obj.get("body", "") or ""
        if body:
            return str(subject), clean_output_text(str(body))
        if obj.get("error"):
            return "", f"ERROR: {obj.get('error')}"
        if obj.get("message"):
            return "", str(obj.get("message"))
        if obj.get("reason"):
            return "", f"ERROR: {obj.get('reason')}"
    if isinstance(output, str) and output.strip():
        return "", clean_output_text(output)
    return "", ""


def get_payload_from_input(input_field: Any) -> Dict[str, Any]:
    """Return the payload from a Langfuse input field."""
    obj = safe_json_loads(input_field)
    if not isinstance(obj, dict):
        return {}
    return obj.get("kwargs", {}).get("payload", {}) or {}


def extract_prompt_text(input_field: Any) -> str:
    """Return the prompt text from a Langfuse input field."""
    obj = safe_json_loads(input_field)
    if not isinstance(obj, dict):
        return ""
    args = obj.get("args", [])
    if args and isinstance(args, list):
        return str(args[0])
    return ""


def extract_between(text: str, start_marker: str, end_marker: str = "---") -> str:
    """Extract text between start_marker and end_marker."""
    if start_marker not in text:
        return ""
    after = text.split(start_marker, 1)[1]
    if end_marker and end_marker in after:
        after = after.split(end_marker, 1)[0]
    return after.strip()


def parse_brief_from_prompt(prompt: str) -> Dict[str, Any]:
    """Best-effort extraction of company + signal summary from cold outreach prompts."""
    brief: Dict[str, Any] = {}
    company_match = re.search(r"at\s+([^\.\n]+?)\.\s*Your goal", prompt)
    if company_match:
        brief["prospect_company"] = company_match.group(1).strip()
    # Pull simple values from the serialized dict embedded in the prompt.
    for key in [
        "prospect_company",
        "job_count",
        "hiring_velocity",
        "confidence",
        "summary",
    ]:
        m = re.search(rf"['\"]{key}['\"]\s*:\s*['\"]?([^,'\"}}\n]+)", prompt)
        if m:
            brief[key] = m.group(1).strip()
    summary_match = re.search(r"['\"]summary['\"]\s*:\s*\\?\"(.+?)\\?\"", prompt)
    if summary_match:
        brief["summary"] = summary_match.group(1).strip()
    return brief


def classify_generated_body(
    body: str, expected_intent: str = ""
) -> Tuple[int, str, str]:
    """Return (label, failure_code, human_tag) for an actual model body."""
    lower = body.lower()
    if not body:
        return 0, "F3.2", "F3.2: Tool Error Unhandled - no prospect-facing output"
    if body.startswith("ERROR:") or "unexpected error" in lower or "not found" in lower:
        return 0, "F3.2", "F3.2: Tool Error Unhandled"
    if "localhost" in lower or "[link]" in lower:
        return 0, "F3.1", "F3.1: Tool Use Error - internal/test link leak"
    if (
        expected_intent == "INTERESTED_BOOK_MEETING"
        and "cal.com" not in lower
        and "book" not in lower
    ):
        return 0, "F2.2", "F2.2: Flawed Logic Flow - missing booking action"
    if any(p in lower for p in BANNED_PHRASES):
        return 0, "F2.3", "F2.3: Tone Drift - banned phrase"
    if "we'll help you ship impactful products faster" in lower:
        return 0, "F2.3", "F2.3: Tone Drift - generic marketing filler"
    return 1, "None", "None"


def corrected_booking_reply(prospect_text: str = "") -> str:
    """Return a corrected reply for a booking request."""
    if "tomorrow" in (prospect_text or "").lower():
        return f"Absolutely. Please choose a time that works for you here: {BOOKING_LINK}. If tomorrow is best, pick any open slot and I’ll make sure the context is attached."
    return f"Absolutely. Please choose a time that works for you here: {BOOKING_LINK}. I’ll include the context for the Tenacious delivery lead."


def corrected_cold_email(brief: Dict[str, Any]) -> str:
    """Return a corrected reply for a cold outreach."""
    company = brief.get("prospect_company") or "your team"
    velocity = str(brief.get("hiring_velocity") or "unknown").lower()
    confidence = str(brief.get("confidence") or "medium").lower()
    if velocity in {"none", "low", "unknown"} or confidence in {"low", "medium"}:
        opener = f"Curious how {company} is balancing roadmap priorities with current engineering capacity."
    else:
        opener = (
            f"Noticed signals that {company} may be expanding engineering capacity."
        )
    return (
        f"{opener} Tenacious helps teams add managed engineering capacity without over-committing before the need is clear. "
        "Would a 15-minute scoping call next week be useful? I can bring a brief view of the signal and where it may or may not point."
    )


def rejected_variant_for(body_type: str, context: Dict[str, Any]) -> str:
    """Return a rejected variant for a given body type and context."""
    if body_type == "booking":
        return "Thanks for letting me know. We help teams ship impactful products faster. Looking forward to connecting soon."
    if body_type == "cold":
        company = context.get("prospect_company") or "your company"
        return (
            f"I noticed {company} is scaling aggressively and probably needs world-class offshore talent. "
            "We can immediately place a bench of A-players and supercharge your roadmap. Quick chat?"
        )
    return (
        "ERROR: internal pipeline failed before a prospect-safe response was produced."
    )


def make_task(
    *,
    pair_id: str,
    label: int,
    prospect_input: str,
    agent_output: str,
    failure_code: str,
    failure_mode_tag: str,
    trace_id: str,
    scenario_type: str,
    subject: str = "",
    source_output_was_actual: bool = False,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create a task from a trace span."""
    task_id = stable_id(pair_id, label, agent_output, trace_id)
    return {
        "task_id": f"trace_{task_id}",
        "pair_id": pair_id,
        "source_mode": "trace-derived",
        "prospect_input": prospect_input,
        "agent_output": agent_output,
        "label": label,
        "failure_code": failure_code,
        "failure_mode_tag": failure_mode_tag,
        "scenario_type": scenario_type,
        "outreach_subject": subject,
        "metadata": {
            "trace_id": trace_id,
            "source": "week10_langfuse_trace",
            "actual_week10_output": source_output_was_actual,
            **(extra_metadata or {}),
        },
    }


def build_trace_tasks(
    input_file: str, output_file: str, max_pairs: Optional[int] = None
) -> List[Dict[str, Any]]:
    """Build tasks from a trace file."""
    traces: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for span in read_jsonl(input_file):
        tid = span.get("traceId")
        if tid:
            traces[tid].append(span)

    tasks: List[Dict[str, Any]] = []

    for tid, spans in sorted(
        traces.items(), key=lambda item: min(s.get("startTime", "") for s in item[1])
    ):
        names = {s.get("name") for s in spans}
        error_spans = [
            s for s in spans if s.get("level") == "ERROR" or s.get("statusMessage")
        ]

        # ---- Handle inbound reply traces ----
        reply_spans = [s for s in spans if s.get("name") == "handle-email-reply"]
        if reply_spans:
            reply = reply_spans[0]
            payload = get_payload_from_input(reply.get("input"))
            data = payload.get("data", {}) if isinstance(payload, dict) else {}
            prospect_text = data.get("text") or ""
            output_obj = safe_json_loads(reply.get("output"))
            pipeline_intent = (
                output_obj.get("intent", "") if isinstance(output_obj, dict) else ""
            )
            status = (
                output_obj.get("status", "") if isinstance(output_obj, dict) else ""
            )
            reason = (
                output_obj.get("reason", "") if isinstance(output_obj, dict) else ""
            )

            # Classification spans can prove the model perceived intent even if the handler failed.
            classify_intents = []
            for s in spans:
                if s.get("name") == "generate_llm_response":
                    out = safe_json_loads(s.get("output"))
                    if isinstance(out, dict) and out.get("intent"):
                        classify_intents.append(out.get("intent"))
            expected_intent = pipeline_intent or (
                classify_intents[-1] if classify_intents else ""
            )

            # Find actual final reply body if one exists.
            generated_bodies: List[Tuple[str, str]] = []
            for s in spans:
                if s.get("name") == "generate_llm_response":
                    subject, body = extract_output_body(s.get("output"))
                    if (
                        body
                        and not body.startswith('{"intent"')
                        and "intent" not in safe_json_loads(s.get("output"))
                    ):
                        # Keep only generated reply/error bodies, not classifier outputs.
                        generated_bodies.append((subject, body))

            has_handler_error = (
                reply.get("level") == "ERROR" or status == "error" or bool(reason)
            )
            if prospect_text or expected_intent:
                pair_id = f"trace_reply_{tid}"
                prompt = (
                    "Prospect replied to Tenacious outreach.\n"
                    f"Prospect reply: {prospect_text or '[missing text]'}\n"
                    f"Classified intent: {expected_intent or 'UNKNOWN'}\n"
                    f"Handler status: {status or reply.get('level', 'UNKNOWN')}\n"
                    "Judge whether the candidate output is the response Kai should send next."
                )

                chosen = (
                    corrected_booking_reply(prospect_text)
                    if expected_intent == "INTERESTED_BOOK_MEETING"
                    else "Thanks for the note. I’ll route this to the right person and avoid sending an automated follow-up until the intent is clear."
                )

                # If the real system produced a weak/failing output, use it as rejected; otherwise create a diagnostic rejected variant.
                actual_subject, actual_body = (
                    generated_bodies[-1] if generated_bodies else ("", "")
                )
                if has_handler_error or not actual_body:
                    rejected = f"ERROR: {reason or reply.get('statusMessage') or 'No prospect-facing booking reply was sent.'}"
                    failure_code = (
                        "F3.2" if reason or reply.get("statusMessage") else "F2.2"
                    )
                    failure_tag = (
                        "F3.2: Tool Error Unhandled"
                        if failure_code == "F3.2"
                        else "F2.2: Flawed Logic Flow - missing booking reply"
                    )
                else:
                    actual_label, failure_code, failure_tag = classify_generated_body(
                        actual_body, expected_intent
                    )
                    if actual_label == 0:
                        rejected = actual_body
                    else:
                        # Keep the successful actual output as chosen, and synthesize a bad contrast.
                        chosen = actual_body
                        rejected = rejected_variant_for("booking", {})
                        failure_code = "F2.2"
                        failure_tag = "F2.2: Flawed Logic Flow - missing booking action"

                tasks.append(
                    make_task(
                        pair_id=pair_id,
                        label=1,
                        prospect_input=prompt,
                        agent_output=chosen,
                        failure_code="None",
                        failure_mode_tag="None",
                        trace_id=tid,
                        scenario_type="reply_followup",
                        subject="Re: Discovery call",
                        source_output_was_actual=chosen
                        in [b for _, b in generated_bodies],
                        extra_metadata={
                            "expected_intent": expected_intent,
                            "handler_status": status,
                        },
                    )
                )
                tasks.append(
                    make_task(
                        pair_id=pair_id,
                        label=0,
                        prospect_input=prompt,
                        agent_output=rejected,
                        failure_code=failure_code,
                        failure_mode_tag=failure_tag,
                        trace_id=tid,
                        scenario_type="reply_followup",
                        subject=actual_subject,
                        source_output_was_actual=rejected
                        in [b for _, b in generated_bodies]
                        or rejected.startswith("ERROR:"),
                        extra_metadata={
                            "expected_intent": expected_intent,
                            "handler_status": status,
                        },
                    )
                )
                continue

        # ---- Handle cold outreach traces ----
        outreach_spans = [
            s
            for s in spans
            if s.get("name") in {"full-outreach-pipeline", "start_outreach_pipeline"}
        ]
        llm_email_spans = []
        for s in spans:
            if s.get("name") == "generate_llm_response":
                prompt = extract_prompt_text(s.get("input"))
                if "short, impactful cold outreach email" in prompt:
                    llm_email_spans.append(s)

        if outreach_spans or llm_email_spans:
            main_span = outreach_spans[0] if outreach_spans else llm_email_spans[0]
            payload = get_payload_from_input(main_span.get("input"))
            company = (
                payload.get("company")
                or main_span.get("metadata", {}).get("company")
                or "Unknown company"
            )
            name = payload.get("name") or "Prospect"
            prompt_text = (
                extract_prompt_text(llm_email_spans[-1].get("input"))
                if llm_email_spans
                else ""
            )
            brief = parse_brief_from_prompt(prompt_text)
            brief.setdefault("prospect_company", company)

            pair_id = f"trace_outreach_{tid}"
            prompt = (
                "Draft or judge a first-touch Tenacious outreach email.\n"
                f"Prospect: {name}\nCompany: {company}\n"
                f"Hiring signal summary: {brief.get('summary') or 'No reliable public signal available.'}\n"
                "The candidate should be concise, grounded, honest about weak signal, and contain one clear ask."
            )
            chosen = corrected_cold_email(brief)

            actual_subject, actual_body = ("", "")
            if llm_email_spans:
                actual_subject, actual_body = extract_output_body(
                    llm_email_spans[-1].get("output")
                )

            if actual_body and not actual_body.startswith("ERROR:"):
                actual_label, failure_code, failure_tag = classify_generated_body(
                    actual_body
                )
                if actual_label == 0:
                    rejected = actual_body
                else:
                    chosen = actual_body
                    rejected = rejected_variant_for("cold", brief)
                    failure_code = "F1.2"
                    failure_tag = (
                        "F1.2: Data Misinterpretation / over-claiming weak signal"
                    )
            else:
                status_msg = (
                    main_span.get("statusMessage")
                    or actual_body
                    or "Pipeline failed before a usable email was generated."
                )
                rejected = f"ERROR: {status_msg}"
                failure_code = "F3.2"
                failure_tag = "F3.2: Tool Error Unhandled"

            tasks.append(
                make_task(
                    pair_id=pair_id,
                    label=1,
                    prospect_input=prompt,
                    agent_output=chosen,
                    failure_code="None",
                    failure_mode_tag="None",
                    trace_id=tid,
                    scenario_type="cold_outreach",
                    subject=actual_subject or "Request: engineering capacity",
                    source_output_was_actual=chosen == actual_body,
                    extra_metadata={"company": company, "brief": brief},
                )
            )
            tasks.append(
                make_task(
                    pair_id=pair_id,
                    label=0,
                    prospect_input=prompt,
                    agent_output=rejected,
                    failure_code=failure_code,
                    failure_mode_tag=failure_tag,
                    trace_id=tid,
                    scenario_type="cold_outreach",
                    subject=actual_subject,
                    source_output_was_actual=rejected == actual_body
                    or rejected.startswith("ERROR:"),
                    extra_metadata={"company": company, "brief": brief},
                )
            )
            continue

        # ---- Handle webhook-only failures as action-failure tasks ----
        webhook_spans = [s for s in spans if s.get("name") == "handle_resend_webhook"]
        if webhook_spans and error_spans:
            s = webhook_spans[0]
            payload = get_payload_from_input(s.get("input"))
            data = payload.get("data", {}) if isinstance(payload, dict) else {}
            prospect_text = data.get("text") or "Incoming Resend webhook"
            pair_id = f"trace_webhook_{tid}"
            prompt = (
                "A Resend webhook arrived for the Tenacious agent.\n"
                f"Webhook text: {prospect_text}\n"
                "Judge whether the candidate safely handles the webhook without crashing or exposing internals."
            )
            tasks.append(
                make_task(
                    pair_id=pair_id,
                    label=1,
                    prospect_input=prompt,
                    agent_output="Acknowledge the event internally, classify the reply if it is prospect-authored, and do not expose any technical error to the prospect.",
                    failure_code="None",
                    failure_mode_tag="None",
                    trace_id=tid,
                    scenario_type="webhook_handling",
                )
            )
            tasks.append(
                make_task(
                    pair_id=pair_id,
                    label=0,
                    prospect_input=prompt,
                    agent_output=f"ERROR: {s.get('statusMessage') or 'Webhook handler crashed.'}",
                    failure_code="F3.3",
                    failure_mode_tag="F3.3: Malformed Output / webhook parsing failure",
                    trace_id=tid,
                    scenario_type="webhook_handling",
                    source_output_was_actual=True,
                )
            )

    # Deduplicate by pair_id + label, keeping first occurrence.
    seen = set()
    deduped: List[Dict[str, Any]] = []
    for task in tasks:
        key = (task["pair_id"], task["label"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(task)

    if max_pairs is not None:
        kept_pair_ids = []
        for task in deduped:
            if task["pair_id"] not in kept_pair_ids:
                kept_pair_ids.append(task["pair_id"])
            if len(kept_pair_ids) >= max_pairs:
                break
        deduped = [t for t in deduped if t["pair_id"] in set(kept_pair_ids)]

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(deduped, f, indent=2, ensure_ascii=False)

    pair_count = len({t["pair_id"] for t in deduped})
    print(
        f"Generated {len(deduped)} trace-derived tasks ({pair_count} pairs) -> {output_file}"
    )
    return deduped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw/llm_traces.jsonl")
    parser.add_argument("--output", default="data/tasks/trace_tasks.json")
    parser.add_argument("--max-pairs", type=int, default=None)
    args = parser.parse_args()
    build_trace_tasks(args.input, args.output, args.max_pairs)


if __name__ == "__main__":
    main()
