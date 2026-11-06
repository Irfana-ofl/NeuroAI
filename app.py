from flask import Flask, render_template, request, jsonify
import urllib.request
import urllib.error
import json
import sqlite3
import time
import re
from pathlib import Path
from statistics import mean

app = Flask(__name__)

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.2:3b"
BASE_DIR = Path(__file__).resolve().parent
DATABASE = BASE_DIR / "neuropulse_metrics.db"

# Thresholds are explicit experimental settings, not learned/calibrated values.
LOW_RISK_THRESHOLD = 25
HIGH_RISK_THRESHOLD = 65
MAX_VERIFY_TOKENS = 1800


# -------------------- Database --------------------

def get_db():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn


def initialize_database():
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY CHECK(id=1),
                total_runs INTEGER DEFAULT 0,
                verified_runs INTEGER DEFAULT 0,
                revisions INTEGER DEFAULT 0,
                abstentions INTEGER DEFAULT 0,
                total_tokens INTEGER DEFAULT 0,
                total_latency REAL DEFAULT 0,
                benchmark_total INTEGER DEFAULT 0,
                benchmark_correct INTEGER DEFAULT 0,
                correction_attempts INTEGER DEFAULT 0,
                benchmark_correction_attempts INTEGER DEFAULT 0,
                benchmark_correction_successes INTEGER DEFAULT 0
            )
        """)
        conn.execute("INSERT OR IGNORE INTO metrics(id) VALUES(1)")

        # Upgrade databases created by earlier NeuroPulse prototypes in place.
        existing = {row[1] for row in conn.execute("PRAGMA table_info(metrics)").fetchall()}
        migrations = {
            "revisions": "INTEGER DEFAULT 0",
            "abstentions": "INTEGER DEFAULT 0",
            "total_tokens": "INTEGER DEFAULT 0",
            "total_latency": "REAL DEFAULT 0",
            "benchmark_correction_attempts": "INTEGER DEFAULT 0",
            "benchmark_correction_successes": "INTEGER DEFAULT 0"
        }
        for column, definition in migrations.items():
            if column not in existing:
                conn.execute(f"ALTER TABLE metrics ADD COLUMN {column} {definition}")

        conn.execute("""
            CREATE TABLE IF NOT EXISTS correction_memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question TEXT NOT NULL,
                corrected_answer TEXT NOT NULL,
                reference_note TEXT DEFAULT '',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS run_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                decision TEXT,
                risk_score REAL,
                verified INTEGER,
                revised INTEGER,
                tokens INTEGER,
                latency REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)


def get_metrics():
    with get_db() as conn:
        row = conn.execute("SELECT * FROM metrics WHERE id=1").fetchone()
    return dict(row)


def record_run(decision, risk_score, verified, revised, tokens, latency):
    with get_db() as conn:
        conn.execute("""
            UPDATE metrics SET
                total_runs=total_runs+1,
                verified_runs=verified_runs+?,
                revisions=revisions+?,
                abstentions=abstentions+?,
                total_tokens=total_tokens+?,
                total_latency=total_latency+?
            WHERE id=1
        """, (int(verified), int(revised), int(decision == "ABSTAIN"),
              int(tokens), float(latency)))
        conn.execute("""
            INSERT INTO run_history(decision,risk_score,verified,revised,tokens,latency)
            VALUES(?,?,?,?,?,?)
        """, (decision, risk_score, int(verified), int(revised), int(tokens), float(latency)))


def aggregate_metrics():
    m = get_metrics()
    total = m["total_runs"]
    verified = m["verified_runs"]
    attempts = m["benchmark_correction_attempts"]
    successes = m["benchmark_correction_successes"]
    benchmark_total = m["benchmark_total"]
    benchmark_correct = m["benchmark_correct"]

    return {
        "accuracy": (f'{benchmark_correct / benchmark_total * 100:.1f}%'
                     if benchmark_total else "Awaiting labelled benchmark"),
        "verification_frequency": (f'{verified / total * 100:.1f}% ({verified}/{total} runs)'
                                   if total else "Awaiting test runs"),
        "correction_success": (f'{successes / attempts * 100:.1f}% ({successes}/{attempts})'
                               if attempts else "Awaiting correction attempts"),
        "total_runs": total,
        "average_tokens": round(m["total_tokens"] / total, 1) if total else 0,
        "average_latency": round(m["total_latency"] / total, 2) if total else 0
    }


initialize_database()


# -------------------- Ollama utilities --------------------

def ollama_generate(prompt, json_mode=False, num_predict=400):
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.2, "num_predict": num_predict}
    }
    if json_mode:
        payload["format"] = "json"

    req = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=180) as response:
        result = json.loads(response.read().decode("utf-8"))

    return {
        "text": result.get("response", "").strip(),
        "prompt_tokens": int(result.get("prompt_eval_count") or 0),
        "output_tokens": int(result.get("eval_count") or 0),
        "total_tokens": int(result.get("prompt_eval_count") or 0) +
                        int(result.get("eval_count") or 0),
        "total_duration_ns": int(result.get("total_duration") or 0),
        "load_duration_ns": int(result.get("load_duration") or 0),
        "eval_duration_ns": int(result.get("eval_duration") or 0)
    }


def extract_json(raw):
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        match = re.search(r"\{.*\}", raw or "", re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                return None
    return None


def add_usage(target, result):
    target["tokens"] += result.get("total_tokens", 0)


def generate_draft(question, memory_context=""):
    prompt = f"""
You are a careful AI assistant. Answer the question directly.
Do not fabricate facts. If evidence is insufficient, state the limitation.
{("Relevant prior correction notes (use only if relevant): " + memory_context) if memory_context else ""}
QUESTION:
{question}
"""
    return ollama_generate(prompt)


def as_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    return str(value).strip().lower() in {"true", "1", "yes"}


def assess_risk(question, draft):
    prompt = f"""
Assess the reliability risk of this draft answer. Do not rewrite it.
Return ONLY JSON:
{{
 "risk_score": 0,
 "confidence_estimate": 0,
 "uncertainty_score": 0,
 "consistency_concern": false,
 "needs_verification": false,
 "should_abstain": false,
 "reason": "brief reason"
}}
Scores are model estimates on 0-100, not calibrated probabilities.
Risk 0 means low apparent risk; 100 means high apparent risk.
Consider ambiguity, missing evidence, internal contradiction, specificity,
and whether the question requires current or specialized facts.
QUESTION: {question}
DRAFT: {draft}
"""
    result = ollama_generate(prompt, json_mode=True, num_predict=220)
    parsed = extract_json(result["text"])
    if not parsed:
        return None, result

    try:
        def score(key):
            return max(0, min(100, float(parsed[key])))
        return {
            "risk": score("risk_score"),
            "confidence": score("confidence_estimate"),
            "uncertainty": score("uncertainty_score"),
            "consistency_concern": as_bool(parsed.get("consistency_concern", False)),
            "needs_verification": as_bool(parsed.get("needs_verification", False)),
            "should_abstain": as_bool(parsed.get("should_abstain", False)),
            "reason": str(parsed.get("reason", "No reason supplied"))
        }, result
    except (KeyError, TypeError, ValueError):
        return None, result


def generate_independent_answer(question):
    return ollama_generate(
        "Answer independently and concisely. Do not assume another answer is correct.\n"
        f"QUESTION: {question}",
        num_predict=300
    )


def compare_answers(question, draft, independent):
    prompt = f"""
Compare two independently generated answers for factual agreement.
Return ONLY JSON:
{{"agreement_score":0,"contradiction":false,"reason":"brief reason"}}
Score is an estimated semantic agreement from 0 to 100, not truth probability.
QUESTION: {question}
ANSWER A: {draft}
ANSWER B: {independent}
"""
    result = ollama_generate(prompt, json_mode=True, num_predict=180)
    parsed = extract_json(result["text"])
    if not parsed:
        return None, result
    try:
        return {
            "agreement": max(0, min(100, float(parsed["agreement_score"]))),
            "contradiction": bool(parsed.get("contradiction", False)),
            "reason": str(parsed.get("reason", ""))
        }, result
    except (KeyError, TypeError, ValueError):
        return None, result


def revise_answer(question, draft, feedback):
    return ollama_generate(f"""
Improve the answer only where needed. Correct unsupported or contradictory claims.
If the available information is insufficient, say so rather than inventing.
QUESTION: {question}
DRAFT: {draft}
VERIFICATION FEEDBACK: {feedback}
Return only the revised answer.
""", num_predict=400)


def final_verification(question, answer, independent):
    prompt = f"""
Check whether the answer is relevant and whether any material contradiction or
unsupported claim is apparent. You cannot guarantee factual truth.
Return ONLY JSON:
{{"verdict":"PASS","reason":"brief reason"}}
Allowed verdicts: PASS, FLAGGED, INCONCLUSIVE.
QUESTION: {question}
FINAL ANSWER: {answer}
INDEPENDENT ANSWER: {independent}
"""
    result = ollama_generate(prompt, json_mode=True, num_predict=180)
    parsed = extract_json(result["text"])
    if not parsed:
        return None, result
    verdict = str(parsed.get("verdict", "")).upper()
    if verdict not in {"PASS", "FLAGGED", "INCONCLUSIVE"}:
        return None, result
    return {"verdict": verdict, "reason": str(parsed.get("reason", ""))}, result


def decide_action(risk, assessment, budget_tokens):
    """Deterministic policy over model-reported signals and configured budget."""
    if assessment is None:
        return "VERIFY", "Risk assessment unavailable; conservative verification."

    if assessment["should_abstain"] and risk >= HIGH_RISK_THRESHOLD:
        return "ABSTAIN", "High risk and insufficient evidence signal."

    if risk >= HIGH_RISK_THRESHOLD:
        return "VERIFY", "High estimated risk exceeds verification threshold."

    if risk >= LOW_RISK_THRESHOLD or assessment["needs_verification"]:
        if budget_tokens < 500:
            return "ABSTAIN", "Risk requires checking but verification budget is insufficient."
        return "VERIFY", "Moderate estimated risk or explicit verification signal."

    return "FINALIZE", "Low estimated risk; optional verification skipped."


def get_memory_context(question, limit=2):
    # Simple keyword-overlap retrieval; this is a transparent prototype memory baseline.
    terms = set(re.findall(r"[a-zA-Z]{4,}", question.lower()))
    if not terms:
        return ""
    with get_db() as conn:
        rows = conn.execute("""
            SELECT question, corrected_answer, reference_note
            FROM correction_memory ORDER BY id DESC LIMIT 100
        """).fetchall()
    scored = []
    for row in rows:
        old_terms = set(re.findall(r"[a-zA-Z]{4,}", row["question"].lower()))
        overlap = len(terms & old_terms) / max(1, len(terms | old_terms))
        if overlap > 0:
            scored.append((overlap, row))
    scored.sort(key=lambda x: x[0], reverse=True)
    return "\n".join(
        f'Prior question: {row["question"]}\nCorrection: {row["corrected_answer"]}\nNote: {row["reference_note"]}'
        for _, row in scored[:limit]
    )


def save_correction(question, answer, note=""):
    with get_db() as conn:
        conn.execute("""
            INSERT INTO correction_memory(question,corrected_answer,reference_note)
            VALUES(?,?,?)
        """, (question, answer, note))


def format_stage_timings(stages):
    return {key: round(value, 3) for key, value in stages.items()}


# -------------------- Routes --------------------

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/ask", methods=["POST"])
def ask():
    started = time.perf_counter()
    data = request.get_json(silent=True) or {}
    question = str(data.get("question", "")).strip()
    budget_tokens = int(data.get("verification_budget_tokens", MAX_VERIFY_TOKENS))
    budget_tokens = max(0, min(budget_tokens, 10000))
    if not question:
        return jsonify({"error": "Please enter a question."}), 400

    events, stages = [], {}
    usage = {"tokens": 0}
    assessment = None
    comparison = None
    final_check = None
    revised = False
    verified = False
    independent = ""
    final_answer = ""
    decision = "VERIFY"

    try:
        memory_context = get_memory_context(question)
        if memory_context:
            events.append("Relevant correction memory retrieved")

        t = time.perf_counter()
        draft_result = generate_draft(question, memory_context)
        stages["draft_generation"] = time.perf_counter() - t
        add_usage(usage, draft_result)
        draft = draft_result["text"]
        if not draft:
            raise RuntimeError("Ollama returned an empty draft.")
        final_answer = draft
        events.append("Draft generated")

        t = time.perf_counter()
        assessment, assess_result = assess_risk(question, draft)
        stages["risk_assessment"] = time.perf_counter() - t
        add_usage(usage, assess_result)
        if assessment:
            risk = assessment["risk"]
            events.append("Risk evidence assessed")
        else:
            risk = 100
            events.append("Risk assessment unavailable; conservative fallback")

        decision, decision_reason = decide_action(risk, assessment, budget_tokens)
        events.append(f"Controller decision: {decision}")

        if decision == "ABSTAIN":
            final_answer = (
                "I cannot provide a sufficiently reliable answer with the "
                "available evidence and verification budget. Please provide "
                "more context or allow a verification pass."
            )

        elif decision == "VERIFY":
            verified = True
            t = time.perf_counter()
            independent_result = generate_independent_answer(question)
            stages["independent_generation"] = time.perf_counter() - t
            add_usage(usage, independent_result)
            independent = independent_result["text"]
            events.append("Independent answer generated")

            t = time.perf_counter()
            comparison, compare_result = compare_answers(question, draft, independent)
            stages["consistency_comparison"] = time.perf_counter() - t
            add_usage(usage, compare_result)

            should_revise = (
                assessment is None
                or assessment["risk"] >= HIGH_RISK_THRESHOLD
                or assessment["should_abstain"]
                or (comparison and (
                    comparison["agreement"] < 65 or comparison["contradiction"]
                ))
            )

            if should_revise:
                t = time.perf_counter()
                feedback = assessment["reason"] if assessment else "Risk assessment failed."
                if comparison:
                    feedback += f" Independent comparison: {comparison['reason']}"
                revision_result = revise_answer(question, draft, feedback)
                stages["revision"] = time.perf_counter() - t
                add_usage(usage, revision_result)
                if revision_result["text"]:
                    final_answer = revision_result["text"]
                    revised = final_answer != draft
                    if revised:
                        decision = "REVISE"
                    events.append("Revision generated" if revised else "Revision unchanged")
                else:
                    events.append("Revision returned no text")

            t = time.perf_counter()
            final_check, final_result = final_verification(
                question, final_answer, independent
            )
            stages["final_verification"] = time.perf_counter() - t
            add_usage(usage, final_result)
            events.append("Final verification completed" if final_check else "Final verification inconclusive")

            if final_check and final_check["verdict"] == "FLAGGED":
                decision = "ABSTAIN"
                final_answer = (
                    "The verification stage flagged a possible unresolved issue. "
                    "I am abstaining rather than presenting this as reliable."
                )

        # If low risk, no second generation/verifier call is performed.
        elapsed = time.perf_counter() - started
        stages["total"] = elapsed
        if assessment:
            confidence = f'{assessment["confidence"]:.0f}% (model-estimated)'
            uncertainty = f'{assessment["uncertainty"]:.0f}% (model-estimated)'
            risk_text = f'{risk:.0f}/100 — {("LOW" if risk < LOW_RISK_THRESHOLD else "MEDIUM" if risk < HIGH_RISK_THRESHOLD else "HIGH")} (heuristic)'
        else:
            confidence = uncertainty = risk_text = "Unavailable"

        consistency_text = (
            f'{comparison["agreement"]:.0f}/100 (estimated agreement)'
            if comparison else ("Not run — adaptive policy skipped verification" if not verified else "Inconclusive")
        )

        if decision == "ABSTAIN":
            outcome = "ABSTAIN — insufficient evidence or unresolved verification flag."
            verdict = "ABSTAIN"
        elif final_check:
            verdict = final_check["verdict"]
            outcome = f'{verdict} — {final_check["reason"]}'
        elif decision == "FINALIZE":
            verdict = "SKIPPED"
            outcome = "Verification skipped by low-risk policy; not independently verified."
        else:
            verdict = "INCONCLUSIVE"
            outcome = "Verification could not establish a reliable result."

        record_run(decision, risk, verified, revised, usage["tokens"], elapsed)
        aggregate = aggregate_metrics()

        return jsonify({
            "answer": final_answer,
            "decision": decision,
            "decision_reason": decision_reason,
            "verification_status": verdict,
            "self_check_reason": final_check["reason"] if final_check else decision_reason,
            "confidence": confidence,
            "error_risk": risk_text,
            "uncertainty": uncertainty,
            "answer_consistency": consistency_text,
            "self_verification": (
                "Independent verification completed" if final_check
                else "Verification skipped by policy" if not verified
                else "Verification attempted; result inconclusive"
            ),
            "verification_outcome": outcome,
            "self_correction": "Revision made" if revised else "No revision made",
            "computational_cost": f'{usage["tokens"]} tokens',
            "correction_memory_used": bool(memory_context),
            "accuracy": aggregate["accuracy"],
            "verification_frequency": aggregate["verification_frequency"],
            "correction_success": aggregate["correction_success"],
            "response_time": f"{elapsed:.2f} s",
            "elapsed_seconds": round(elapsed, 2),
            "events": events + ["Evidence metrics updated"],
            "stage_timings": format_stage_timings(stages),
            "aggregate_metrics": aggregate,
            "risk_score": risk,
            "verified": verified,
            "revised": revised
        })

    except Exception as error:
        print("NEUROPULSE ERROR:", repr(error))
        return jsonify({
            "error": str(error),
            "message": "Check that Ollama is running and the selected model is installed."
        }), 500


def judge_answer(question, answer, reference):
    prompt = f"""
Compare the response with the reference for semantic correctness.
Return ONLY JSON: {{"score":0,"reason":"brief reason"}}
Score 0-100. Penalize contradictions and unsupported claims.
QUESTION: {question}
REFERENCE: {reference}
RESPONSE: {answer}
"""
    result = ollama_generate(prompt, json_mode=True, num_predict=160)
    parsed = extract_json(result["text"])
    if not parsed:
        return None, result
    try:
        return {"score": max(0, min(100, float(parsed["score"]))),
                "reason": str(parsed.get("reason", ""))}, result
    except (KeyError, TypeError, ValueError):
        return None, result


@app.route("/benchmark", methods=["POST"])
def benchmark():
    data = request.get_json(silent=True) or {}
    items = data.get("items", [])
    if not isinstance(items, list) or not items or len(items) > 100:
        return jsonify({"error": "Provide 1-100 labelled benchmark items."}), 400

    results, correct, attempts, successes = [], 0, 0, 0
    try:
        for item in items:
            question = str(item.get("question", "")).strip()
            reference = str(item.get("reference_answer", "")).strip()
            if not question or not reference:
                continue

            # Run the same adaptive policy used by /ask.
            draft_result = generate_draft(question)
            draft = draft_result["text"]
            assessment, _ = assess_risk(question, draft)
            risk = assessment["risk"] if assessment else 100
            action, _ = decide_action(risk, assessment, MAX_VERIFY_TOKENS)
            final_answer = draft
            revised = False

            if action == "VERIFY":
                independent = generate_independent_answer(question)["text"]
                comparison, _ = compare_answers(question, draft, independent)
                needs_revision = (
                    assessment is None or risk >= HIGH_RISK_THRESHOLD or
                    (comparison and (comparison["agreement"] < 65 or comparison["contradiction"]))
                )
                if needs_revision:
                    attempts += 1
                    rev = revise_answer(question, draft, assessment["reason"] if assessment else "Uncertain")
                    if rev["text"]:
                        final_answer = rev["text"]
                        revised = final_answer != draft

            score, _ = judge_answer(question, final_answer, reference)
            before, _ = judge_answer(question, draft, reference)
            if not score or not before:
                continue

            final_correct = score["score"] >= 80
            initial_correct = before["score"] >= 80
            correct += int(final_correct)
            if revised and not initial_correct and final_correct:
                successes += 1

            results.append({
                "question": question,
                "initial_score": round(before["score"], 1),
                "final_score": round(score["score"], 1),
                "decision": action,
                "revision_made": revised,
                "final_correct": final_correct
            })

            # Only add an approved, reference-grounded correction to memory.
            if final_correct and revised:
                save_correction(question, final_answer, "Passed reference-based LLM-judge threshold; manual review recommended.")

        total = len(results)
        if not total:
            return jsonify({"error": "No benchmark items were successfully scored."}), 400

        with get_db() as conn:
            conn.execute("""
                UPDATE metrics SET
                    benchmark_total=benchmark_total+?,
                    benchmark_correct=benchmark_correct+?,
                    benchmark_correction_attempts=benchmark_correction_attempts+?,
                    benchmark_correction_successes=benchmark_correction_successes+?
                WHERE id=1
            """, (total, correct, attempts, successes))

        return jsonify({
            "benchmark_questions_evaluated": total,
            "correct_answers": correct,
            "benchmark_accuracy": round(correct / total * 100, 2),
            "correction_attempts": attempts,
            "successful_corrections": successes,
            "correction_success_rate": round(successes / attempts * 100, 2) if attempts else None,
            "results": results,
            "aggregate_metrics": aggregate_metrics(),
            "evaluation_note": "LLM-judged benchmark estimate; use human/adjudicated labels for publication-grade ground truth."
        })
    except Exception as error:
        print("BENCHMARK ERROR:", repr(error))
        return jsonify({"error": str(error)}), 500


@app.route("/capabilities", methods=["POST"])
def capabilities():
    """
    Four small, user-supplied labelled test groups.
    Expected item shapes:
      intelligence/reasoning: {question, reference_answer}
      learning: {question, reference_answer, correction_context}
      decision: {question, expected_decision}
    """
    data = request.get_json(silent=True) or {}
    groups = data.get("tests", {})
    names = ["intelligence", "reasoning", "learning", "decision"]
    output = {}

    try:
        for name in names:
            items = groups.get(name, [])
            if not isinstance(items, list) or not items:
                output[name] = {"score": None, "n": 0, "status": "No test items supplied"}
                continue

            scores = []
            details = []
            for item in items[:50]:
                question = str(item.get("question", "")).strip()
                if not question:
                    continue

                if name in ("intelligence", "reasoning"):
                    reference = str(item.get("reference_answer", "")).strip()
                    if not reference:
                        continue
                    answer = generate_draft(question)["text"]
                    judged, _ = judge_answer(question, answer, reference)
                    if judged:
                        scores.append(judged["score"])
                        details.append({"score": round(judged["score"], 1)})

                elif name == "learning":
                    reference = str(item.get("reference_answer", "")).strip()
                    context = str(item.get("correction_context", "")).strip()
                    if not reference or not context:
                        continue
                    baseline = generate_draft(question)["text"]
                    memory_answer = generate_draft(question, context)["text"]
                    base_score, _ = judge_answer(question, baseline, reference)
                    mem_score, _ = judge_answer(question, memory_answer, reference)
                    if base_score and mem_score:
                        delta = mem_score["score"] - base_score["score"]
                        scores.append(max(0, min(100, 50 + delta)))
                        details.append({
                            "baseline_score": round(base_score["score"], 1),
                            "memory_score": round(mem_score["score"], 1),
                            "improvement": round(delta, 1)
                        })

                else:  # decision quality
                    expected = str(item.get("expected_decision", "")).upper()
                    if expected not in {"FINALIZE", "VERIFY", "REVISE", "ABSTAIN"}:
                        continue
                    draft = generate_draft(question)["text"]
                    assessment, _ = assess_risk(question, draft)
                    risk = assessment["risk"] if assessment else 100
                    predicted, _ = decide_action(risk, assessment, MAX_VERIFY_TOKENS)
                    # REVISE is evaluated as a VERIFY-stage decision that subsequently revises.
                    hit = predicted == expected or (expected == "REVISE" and predicted == "VERIFY")
                    scores.append(100.0 if hit else 0.0)
                    details.append({"expected": expected, "predicted": predicted, "correct": hit})

            output[name] = {
                "score": round(mean(scores), 1) if scores else None,
                "n": len(scores),
                "status": "Evaluated" if scores else "No valid labelled items",
                "details": details
            }

        return jsonify({
            "capabilities": output,
            "method_note": "Capability scores are task-specific prototype evaluation results, not a general intelligence score. Learning is a correction-context A/B comparison."
        })
    except Exception as error:
        print("CAPABILITY ERROR:", repr(error))
        return jsonify({"error": str(error)}), 500


@app.route("/experiment", methods=["POST"])
@app.route("/experiment", methods=["POST"])
def experiment():
    """
    Compare three strategies on the same labelled items.

    baseline:
        Draft generation only.

    always_verify:
        Draft + independent generation + comparison
        + conditional revision.

    adaptive:
        Draft + risk assessment + conditional verification
        + conditional revision.

    Strategy token and latency measurements exclude
    evaluator/judge overhead for a fair comparison.
    """
    data = request.get_json(silent=True) or {}
    items = data.get("items", [])

    if not isinstance(items, list) or not items or len(items) > 50:
        return jsonify({
            "error": "Provide 1-50 labelled experiment items."
        }), 400

    totals = {
        "baseline": {
            "correct": 0,
            "tokens": 0,
            "verified": 0,
            "latency": []
        },
        "always_verify": {
            "correct": 0,
            "tokens": 0,
            "verified": 0,
            "latency": []
        },
        "adaptive": {
            "correct": 0,
            "tokens": 0,
            "verified": 0,
            "latency": []
        }
    }

    results = []
    evaluation_overhead_tokens = 0
    skipped_items = 0

    try:
        for item in items:
            question = str(item.get("question", "")).strip()
            reference = str(item.get("reference_answer", "")).strip()

            if not question or not reference:
                skipped_items += 1
                continue

            # ----------------------------------
            # 1. SHARED DRAFT
            # ----------------------------------
            draft_start = time.perf_counter()

            draft_res = generate_draft(question)
            draft_elapsed = time.perf_counter() - draft_start

            draft = draft_res["text"]
            draft_tokens = draft_res["total_tokens"]

            if not draft:
                skipped_items += 1
                continue

            row = {"question": question}

            # ----------------------------------
            # 2. BASELINE: DRAFT ONLY
            # ----------------------------------
            baseline_answer = draft
            baseline_tokens = draft_tokens
            baseline_latency = draft_elapsed

            baseline_judge, baseline_judge_result = judge_answer(
                question,
                baseline_answer,
                reference
            )

            # ----------------------------------
            # 3. ALWAYS-VERIFY
            # ----------------------------------
            always_start = time.perf_counter()

            ind = generate_independent_answer(question)

            cmp, cmp_result = compare_answers(
                question,
                draft,
                ind["text"]
            )

            always_answer = draft

            always_tokens = (
                draft_tokens
                + ind["total_tokens"]
                + cmp_result["total_tokens"]
            )

            always_revised = False

            if cmp and (
                cmp["agreement"] < 65
                or cmp["contradiction"]
            ):
                rev = revise_answer(
                    question,
                    draft,
                    cmp["reason"]
                )

                always_tokens += rev["total_tokens"]

                if rev["text"]:
                    always_answer = rev["text"]
                    always_revised = always_answer != draft

            # Stop strategy timer BEFORE evaluator call.
            always_processing_elapsed = (
                time.perf_counter() - always_start
            )

            always_latency = (
                draft_elapsed + always_processing_elapsed
            )

            always_judge, always_judge_result = judge_answer(
                question,
                always_answer,
                reference
            )

            # ----------------------------------
            # 4. ADAPTIVE NEUROPULSE
            # ----------------------------------
            adaptive_start = time.perf_counter()

            assessment, assess_result = assess_risk(
                question,
                draft
            )

            risk = assessment["risk"] if assessment else 100

            action, decision_reason = decide_action(
                risk,
                assessment,
                MAX_VERIFY_TOKENS
            )

            adaptive_answer = draft
            adaptive_verified = action == "VERIFY"
            adaptive_revised = False

            # Risk assessment is part of adaptive strategy cost.
            adaptive_tokens = (
                draft_tokens + assess_result["total_tokens"]
            )

            if adaptive_verified:
                ind2 = generate_independent_answer(question)

                cmp2, cmp2_result = compare_answers(
                    question,
                    draft,
                    ind2["text"]
                )

                # Count each model call exactly once.
                adaptive_tokens += (
                    ind2["total_tokens"]
                    + cmp2_result["total_tokens"]
                )

                needs_revision = (
                    assessment is None
                    or risk >= HIGH_RISK_THRESHOLD
                    or (
                        cmp2 and (
                            cmp2["agreement"] < 65
                            or cmp2["contradiction"]
                        )
                    )
                )

                if needs_revision:
                    feedback = (
                        assessment["reason"]
                        if assessment
                        else "Risk assessment unavailable."
                    )

                    if cmp2:
                        feedback += (
                            " Independent comparison: "
                            + cmp2["reason"]
                        )

                    rev2 = revise_answer(
                        question,
                        draft,
                        feedback
                    )

                    adaptive_tokens += rev2["total_tokens"]

                    if rev2["text"]:
                        adaptive_answer = rev2["text"]
                        adaptive_revised = (
                            adaptive_answer != draft
                        )

            # Stop strategy timer BEFORE evaluator call.
            adaptive_processing_elapsed = (
                time.perf_counter() - adaptive_start
            )

            adaptive_latency = (
                draft_elapsed + adaptive_processing_elapsed
            )

            adaptive_judge, adaptive_judge_result = judge_answer(
                question,
                adaptive_answer,
                reference
            )

            # ----------------------------------
            # 5. KEEP ONLY FULLY SCORED ITEMS
            # ----------------------------------
            if not all([
                baseline_judge,
                always_judge,
                adaptive_judge
            ]):
                skipped_items += 1
                continue

            # Evaluator tokens are recorded separately.
            evaluation_overhead_tokens += (
                baseline_judge_result["total_tokens"]
                + always_judge_result["total_tokens"]
                + adaptive_judge_result["total_tokens"]
            )

            baseline_correct = baseline_judge["score"] >= 80
            always_correct = always_judge["score"] >= 80
            adaptive_correct = adaptive_judge["score"] >= 80

            totals["baseline"]["correct"] += int(
                baseline_correct
            )
            totals["baseline"]["tokens"] += baseline_tokens
            totals["baseline"]["latency"].append(
                baseline_latency
            )

            totals["always_verify"]["correct"] += int(
                always_correct
            )
            totals["always_verify"]["tokens"] += always_tokens
            totals["always_verify"]["verified"] += 1
            totals["always_verify"]["latency"].append(
                always_latency
            )

            totals["adaptive"]["correct"] += int(
                adaptive_correct
            )
            totals["adaptive"]["tokens"] += adaptive_tokens
            totals["adaptive"]["verified"] += int(
                adaptive_verified
            )
            totals["adaptive"]["latency"].append(
                adaptive_latency
            )

            row.update({
                "baseline_score": round(
                    baseline_judge["score"], 1
                ),
                "always_verify_score": round(
                    always_judge["score"], 1
                ),
                "adaptive_score": round(
                    adaptive_judge["score"], 1
                ),
                "baseline_correct": baseline_correct,
                "always_verify_correct": always_correct,
                "adaptive_correct": adaptive_correct,
                "adaptive_decision": action,
                "risk_score": round(risk, 1),
                "always_revision_made": always_revised,
                "adaptive_revision_made": adaptive_revised
            })

            results.append(row)

        # ----------------------------------
        # 6. FINAL SUMMARY
        # ----------------------------------
        n = len(results)

        if not n:
            return jsonify({
                "error": (
                    "No items could be evaluated by all three "
                    "strategies. Check the model and evaluator."
                ),
                "skipped_items": skipped_items
            }), 400

        summary = {}

        for name, values in totals.items():
            summary[name] = {
                "accuracy": round(
                    values["correct"] / n * 100, 1
                ),
                "verification_frequency": round(
                    values["verified"] / n * 100, 1
                ),
                "average_tokens": round(
                    values["tokens"] / n, 1
                ),
                "average_latency_seconds": round(
                    mean(values["latency"]), 2
                ) if values["latency"] else None
            }

        return jsonify({
            "n": n,
            "skipped_items": skipped_items,
            "strategies": summary,
            "results": results,
            "evaluation_overhead_tokens": (
                evaluation_overhead_tokens
            ),
            "accuracy_threshold": 80,
            "note": (
                "Accuracy values are LLM-judge estimates, not "
                "human-adjudicated ground truth. Strategy token "
                "and latency measurements exclude evaluator "
                "calls. Shared draft cost is included in all "
                "three strategies. Each strategy call is counted "
                "once."
            )
        })

    except Exception as error:
        print("EXPERIMENT ERROR:", repr(error))
        return jsonify({"error": str(error)}), 500

@app.route("/metrics", methods=["GET"])
def metrics():
    m = get_metrics()
    with get_db() as conn:
        recent = conn.execute("""
            SELECT decision,risk_score,verified,revised,tokens,latency,created_at
            FROM run_history ORDER BY id DESC LIMIT 30
        """).fetchall()
        memory_count = conn.execute("SELECT COUNT(*) AS n FROM correction_memory").fetchone()["n"]
    return jsonify({
        "statistics": m,
        "display_metrics": aggregate_metrics(),
        "recent_runs": [dict(r) for r in recent],
        "correction_memory_items": memory_count
    })


if __name__ == "__main__":
    print("\n====================================")
    print("       NEUROPULSE AI")
    print(" Adaptive Verification Framework")
    print("====================================")
    print("Model:", MODEL)
    print("Database:", DATABASE)
    print("Dashboard: http://127.0.0.1:5000")
    print("====================================\n")
    app.run(host="127.0.0.1", port=5000, debug=True, use_reloader=False)
