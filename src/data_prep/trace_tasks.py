# src/data_prep/trace_tasks.py
import json
from collections import defaultdict

def build_trace_tasks(input_file, output_file):
    traces_by_id = defaultdict(dict)
    tasks = []

    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                t = json.loads(line)
                tid = t.get("traceId")
                name = t.get("name")
                if not tid or not name:
                    continue

                inner_input, inner_output = {}, {}
                if t.get("input"):
                    try:
                        inner_input = json.loads(t["input"])
                    except: pass
                if t.get("output"):
                    try:
                        inner_output = json.loads(t["output"])
                    except: pass

                # Prospect inbound email
                if name == "handle-email-reply":
                    prospect_text = inner_input.get("kwargs", {}).get("payload", {}).get("data", {}).get("text")
                    intent = inner_output.get("intent", "UNKNOWN")
                    if prospect_text:
                        traces_by_id[tid]["prospect"] = prospect_text
                        traces_by_id[tid]["intent"] = intent
                        print(f"[{tid}] Prospect:", prospect_text, "Intent:", intent)

                # Agent reply OR outreach email
                if name == "generate_llm_response":
                    subject = inner_output.get("subject")
                    body = inner_output.get("body")
                    if subject and body:
                        traces_by_id[tid]["outreach_subject"] = subject
                        traces_by_id[tid]["outreach_body"] = body
                        print(f"[{tid}] Outreach email subject:", subject)
                        print(f"[{tid}] Outreach email body:", body)
                    elif body:
                        traces_by_id[tid]["agent"] = body
                        print(f"[{tid}] Agent reply:", body)

            except Exception as e:
                print("Error parsing line:", e)

    # Build tasks even if only outreach or prospect exists
    for tid, vals in traces_by_id.items():
        task = {
            "prospect_input": vals.get("prospect"),
            "agent_output": vals.get("agent"),
            "outreach_subject": vals.get("outreach_subject"),
            "outreach_body": vals.get("outreach_body"),
            "label": 1 if vals.get("intent") == "INTERESTED_BOOK_MEETING" else 0,
            "failure_mode_tag": "Trace-derived",
            "metadata": f"TraceId {tid}"
        }
        # Only add if at least one field is present
        if any([task["prospect_input"], task["agent_output"], task["outreach_body"]]):
            tasks.append(task)
            print(">>> Added task:", task)

    print(f"\nTotal tasks created: {len(tasks)}")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(tasks, f, indent=2)

if __name__ == "__main__":
    build_trace_tasks("data/raw/llm_traces.jsonl", "data/tasks/trace_tasks.json")
