#!/usr/bin/env python3
import html
import json
from collections import defaultdict
from pathlib import Path

from transformers import AutoTokenizer


REPO_ROOT = Path("/work/leotsia0416/projects/SDAR")
DEMO_ROOT = Path("/work/leotsia0416/DRPO-demo")
MODEL_PATH = REPO_ROOT / "checkpoint/grpo_from_remask_head_ref_only_90861/checkpoint-450"
EVENT_TRACE = (
    REPO_ROOT
    / "outputs/math500_grpo90861_ckpt450_850_950_rt050_bs16_8gpu_20260614_160254/checkpoint-450/rt0_50/remask_event_trace.jsonl"
)
VISUAL_CASES = REPO_ROOT / "outputs/remask_visual_cases/math500_ckpt450_rt050/math500_remask_visual_cases.json"
TARGET_HTML = DEMO_ROOT / "index.html"
LOCAL_COPY = REPO_ROOT / "outputs/remask_visual_cases/key_sentence_inline_edits.html"

SELECTED_CASES = [
    "math-500_404",  # find -> subtract
    "math-500_33",   # 8-1 -> 7-1
    "math-500_428",  # 1/3 -> 1/6
    "math-500_444",  # x -> y
]

SYNTHETIC_CHANGED_REMASKS_PER_CASE = 10
SYNTHETIC_REPLACEMENT_TEXTS = [
    ":",
    ";",
    ",",
    ".",
    " then",
    " so",
    " therefore",
    " 2",
    " 3",
    " +",
    " -",
    " =",
    " the",
    " a",
]


def decode(tokenizer, token_ids):
    return tokenizer.decode(token_ids, skip_special_tokens=False)


def esc(text):
    return html.escape(text, quote=False).replace("&lt;|MASK|&gt;", '<span class="mask">MASK</span>')


def prompt_key(record_input):
    return json.dumps(record_input, ensure_ascii=False, sort_keys=True)


def render_decode_block(tokenizer, token_ids, start, end):
    return (
        esc(decode(tokenizer, token_ids[:start]))
        + '<span class="decode-block">'
        + esc(decode(tokenizer, token_ids[start:end]))
        + "</span>"
        + esc(decode(tokenizer, token_ids[end:]))
    )


def render_remask_window(tokenizer, token_ids, prompt_length, window_start, window_end, highlight_positions):
    local_window_start = max(0, window_start - prompt_length)
    local_window_end = max(local_window_start, window_end - prompt_length)
    local_positions = sorted(
        pos - prompt_length
        for pos in highlight_positions
        if local_window_start <= pos - prompt_length < local_window_end
    )

    parts = [esc(decode(tokenizer, token_ids[:local_window_start])), '<span class="remask-window">']
    cursor = local_window_start
    for local_pos in local_positions:
        parts.append(esc(decode(tokenizer, token_ids[cursor:local_pos])))
        parts.append('<span class="decode-block">')
        parts.append(esc(decode(tokenizer, token_ids[local_pos : local_pos + 1])))
        parts.append("</span>")
        cursor = local_pos + 1
    parts.append(esc(decode(tokenizer, token_ids[cursor:local_window_end])))
    parts.append("</span>")
    parts.append(esc(decode(tokenizer, token_ids[local_window_end:])))
    return "".join(parts)


def render_old_new_window(tokenizer, before_ids, after_ids, record):
    prompt_length = record["prompt_length"]
    window_start = record["window_token_start"]
    window_end = record["window_token_end"]
    positions = record.get("remasked_positions") or []
    if not positions:
        return render_remask_window(tokenizer, after_ids, prompt_length, window_start, window_end, [])

    local_window_start = window_start - prompt_length
    local_window_end = window_end - prompt_length
    local_pos = positions[0] - prompt_length

    parts = [esc(decode(tokenizer, after_ids[:local_window_start])), '<span class="remask-window">']
    parts.append(esc(decode(tokenizer, after_ids[local_window_start:local_pos])))
    parts.append('<span class="old">')
    parts.append(esc(decode(tokenizer, before_ids[local_pos : local_pos + 1])))
    parts.append("</span> ")
    parts.append('<span class="decode-block"><span class="new">')
    parts.append(esc(decode(tokenizer, after_ids[local_pos : local_pos + 1])))
    parts.append("</span></span>")
    parts.append(esc(decode(tokenizer, after_ids[local_pos + 1 : local_window_end])))
    parts.append("</span>")
    parts.append(esc(decode(tokenizer, after_ids[local_window_end:])))
    return "".join(parts)


def single_token_replacements(tokenizer):
    replacements = []
    for text in SYNTHETIC_REPLACEMENT_TEXTS:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if len(token_ids) == 1:
            replacements.append(token_ids[0])
    if not replacements:
        raise RuntimeError("No single-token synthetic replacements are available for this tokenizer.")
    return replacements


def synthesize_changed_before_ids(tokenizer, before_ids, after_ids, record, replacement_ids, offset):
    positions = record.get("remasked_positions") or []
    if not positions:
        return before_ids

    local_pos = positions[0] - record["prompt_length"]
    if local_pos < 0 or local_pos >= min(len(before_ids), len(after_ids)):
        return before_ids

    refilled_id = after_ids[local_pos]
    for step in range(len(replacement_ids)):
        replacement_id = replacement_ids[(offset + step) % len(replacement_ids)]
        if replacement_id != refilled_id:
            patched = list(before_ids)
            patched[local_pos] = replacement_id
            return patched
    return before_ids


def load_cases():
    payload = json.loads(VISUAL_CASES.read_text(encoding="utf-8"))
    cases = {case["example_abbr"]: case for case in payload["cases"]}
    selected = [cases[name] for name in SELECTED_CASES if name in cases]
    missing = [name for name in SELECTED_CASES if name not in cases]
    if missing:
        raise RuntimeError(f"Missing visual cases: {missing}")
    return selected


def prompt_needle(case):
    base = case["prompt"].split(" Solve the problem step by step", 1)[0]
    return " ".join(base.split())[:60]


def load_records_by_prompt(cases):
    needles = {case["example_abbr"]: prompt_needle(case) for case in cases}
    prompt_to_records = defaultdict(list)
    abbr_to_prompt = {}
    with EVENT_TRACE.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            prompt = (record.get("input") or [{}])[0].get("prompt", "")
            normalized_prompt = " ".join(prompt.split())
            prompt_to_records[prompt].append(record)
            for abbr, needle in needles.items():
                if needle and needle in normalized_prompt:
                    abbr_to_prompt[abbr] = prompt

    return {
        abbr: prompt_to_records[prompt]
        for abbr, prompt in abbr_to_prompt.items()
    }


def build_steps(tokenizer, records):
    steps = [{"title": "Step 00 / Empty answer", "focus": '<span class="empty-token">0 generated tokens</span>'}]
    prev_after = []
    visible_step = 1
    replacement_ids = single_token_replacements(tokenizer)
    synthetic_count = 0

    for record in records:
        before_ids = record["generated_before_token_ids"]
        original_after_ids = record["generated_after_token_ids"]
        after_ids = original_after_ids

        if before_ids[: len(prev_after)] == prev_after and len(before_ids) > len(prev_after):
            start = len(prev_after)
            end = len(before_ids)
            steps.append(
                {
                    "title": (
                        f"Step {visible_step:02d} / block {record['block_idx']} decode "
                        f"{end - start} tokenizer tokens"
                    ),
                    "focus": render_decode_block(tokenizer, before_ids, start, end),
                }
            )
            visible_step += 1

        changed_by_remask = before_ids != after_ids
        synthetic_change = False
        if record.get("triggered") and not changed_by_remask and synthetic_count < SYNTHETIC_CHANGED_REMASKS_PER_CASE:
            before_ids = synthesize_changed_before_ids(
                tokenizer,
                before_ids,
                after_ids,
                record,
                replacement_ids,
                synthetic_count,
            )
            synthetic_change = before_ids != record["generated_before_token_ids"]
            if synthetic_change:
                synthetic_count += 1
                changed_by_remask = True

        if record.get("triggered") and changed_by_remask:
            positions = record.get("remasked_positions") or []
            demo_note = " (demo token swap)" if synthetic_change else ""
            steps.append(
                {
                    "title": f"Step {visible_step:02d} / block {record['block_idx']} remask phase 1: target{demo_note}",
                    "focus": render_remask_window(
                        tokenizer,
                        before_ids,
                        record["prompt_length"],
                        record["window_token_start"],
                        record["window_token_end"],
                        positions,
                    ),
                }
            )
            visible_step += 1

            masked_ids = record.get("generated_with_masks_token_ids")
            if masked_ids:
                steps.append(
                    {
                        "title": f"Step {visible_step:02d} / block {record['block_idx']} remask phase 2: MASK{demo_note}",
                        "focus": render_remask_window(
                            tokenizer,
                            masked_ids,
                            record["prompt_length"],
                            record["window_token_start"],
                            record["window_token_end"],
                            positions,
                        ),
                    }
                )
                visible_step += 1

            steps.append(
                {
                    "title": f"Step {visible_step:02d} / block {record['block_idx']} remask phase 3: refill{demo_note}",
                    "focus": render_old_new_window(tokenizer, before_ids, after_ids, record),
                }
            )
            visible_step += 1

            steps.append(
                {
                    "title": f"Step {visible_step:02d} / block {record['block_idx']} remask phase 4: commit{demo_note}",
                    "focus": render_remask_window(
                        tokenizer,
                        after_ids,
                        record["prompt_length"],
                        record["window_token_start"],
                        record["window_token_end"],
                        positions,
                    ),
                }
            )
            visible_step += 1

        prev_after = original_after_ids

    return steps


def short_prompt(case):
    prompt = case["prompt"].replace(" Solve the problem step by step", "\nSolve the problem step by step")
    return prompt[:360] + ("..." if len(prompt) > 360 else "")


def build_html(case_payloads):
    data_json = json.dumps(case_payloads, ensure_ascii=False)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>DRPO Remasking Trace Demo</title>
  <style>
    :root {{
      --bg: #f6f1e8;
      --ink: #26231f;
      --muted: #736b60;
      --card: #fffaf1;
      --line: #ded2bf;
      --old-bg: #f7d8d5;
      --old-ink: #8d312b;
      --new-bg: #c9f2ce;
      --new-ink: #155b25;
      --mask-bg: #efe4ff;
      --mask-ink: #5c3f8d;
      --window-bg: rgba(255, 255, 255, 0.38);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background:
        radial-gradient(circle at 14% 8%, rgba(255, 206, 146, 0.35), transparent 30rem),
        radial-gradient(circle at 86% 16%, rgba(145, 190, 255, 0.25), transparent 28rem),
        var(--bg);
      color: var(--ink);
      font-family: ui-serif, Georgia, "Times New Roman", serif;
      line-height: 1.55;
    }}
    main {{
      width: min(1180px, calc(100vw - 32px));
      margin: 38px auto;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: clamp(32px, 5vw, 56px);
      line-height: 1.02;
      letter-spacing: -0.04em;
    }}
    .subtitle {{
      margin: 0 0 22px;
      color: var(--muted);
      font-size: 18px;
    }}
    .tabs {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-bottom: 18px;
    }}
    .tab {{
      appearance: none;
      border: 1px solid #171717;
      border-radius: 999px;
      background: transparent;
      color: #171717;
      cursor: pointer;
      font: 700 13px/1 ui-sans-serif, system-ui, sans-serif;
      padding: 10px 13px;
    }}
    .tab.active {{
      background: #171717;
      color: #fffaf1;
    }}
    .card {{
      border: 1px solid var(--line);
      border-radius: 22px;
      background: color-mix(in srgb, var(--card) 92%, white);
      box-shadow: 0 18px 40px rgba(76, 54, 32, 0.08);
      padding: 22px;
      margin-bottom: 18px;
    }}
    .case-card {{ display: none; }}
    .case-card.active {{ display: block; }}
    .meta {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      align-items: center;
      margin-bottom: 14px;
      color: var(--muted);
      font: 13px/1.2 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }}
    .pill {{
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 5px 9px;
      background: rgba(255, 255, 255, 0.42);
    }}
    .prompt {{
      margin: 0 0 16px;
      color: var(--muted);
      font-size: 14px;
      white-space: pre-wrap;
    }}
    .player {{ display: grid; gap: 18px; }}
    .viewport {{
      position: relative;
      min-height: 360px;
      max-height: 62vh;
      overflow: auto;
      border: 2px solid #171717;
      border-radius: 20px;
      background:
        linear-gradient(135deg, rgba(255, 255, 255, 0.72), rgba(255, 255, 255, 0.28)),
        var(--window-bg);
      padding: 44px 22px 22px;
    }}
    .viewport::before {{
      content: "current decoding block";
      position: absolute;
      top: 12px;
      right: 18px;
      border: 2px solid #171717;
      border-radius: 999px;
      background: var(--card);
      color: #171717;
      padding: 4px 10px;
      font: 11px/1 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }}
    .focus-line {{
      margin: 0;
      font-size: 15px;
      line-height: 1.72;
      white-space: pre-wrap;
    }}
    .empty-token {{ color: var(--muted); font-style: italic; }}
    .decode-block {{
      display: inline;
      border: 3px solid #111;
      border-radius: 12px;
      padding: 0.06em 0.16em 0.12em;
      box-decoration-break: clone;
      -webkit-box-decoration-break: clone;
      background: rgba(255, 255, 255, 0.55);
      box-shadow: 0 0 0 4px rgba(17, 17, 17, 0.05);
    }}
    .remask-window {{
      background: rgba(255, 245, 190, 0.65);
      border-radius: 10px;
      padding: 0.04em 0.18em;
      box-decoration-break: clone;
      -webkit-box-decoration-break: clone;
    }}
    .old {{
      color: var(--old-ink);
      background: var(--old-bg);
      border-radius: 8px;
      padding: 0.04em 0.18em;
      text-decoration: line-through;
      text-decoration-thickness: 0.11em;
    }}
    .new {{
      color: var(--new-ink);
      background: var(--new-bg);
      border-radius: 8px;
      padding: 0.04em 0.22em;
      font-weight: 700;
    }}
    .mask {{
      color: var(--mask-ink);
      background: var(--mask-bg);
      border-radius: 8px;
      padding: 0.04em 0.2em;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 0.78em;
    }}
    .player-status {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      align-items: center;
      justify-content: space-between;
    }}
    .step-title {{
      margin: 0;
      color: var(--muted);
      font: 13px/1.4 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}
    .controls {{ display: flex; gap: 10px; }}
    .control-button {{
      appearance: none;
      border: 1px solid #171717;
      border-radius: 999px;
      background: #171717;
      color: #fffaf1;
      cursor: pointer;
      font: 700 14px/1 ui-sans-serif, system-ui, sans-serif;
      padding: 11px 16px;
    }}
    .control-button.secondary {{
      background: transparent;
      color: #171717;
    }}
    .dots {{
      display: flex;
      flex-wrap: wrap;
      gap: 7px;
      margin-top: -4px;
    }}
    .dot {{
      width: 7px;
      height: 7px;
      border-radius: 999px;
      background: var(--line);
      cursor: pointer;
    }}
    .dot.active {{ background: #171717; }}
    .note {{
      margin: 12px 0 0;
      color: var(--muted);
      font-size: 15px;
    }}
  </style>
</head>
<body>
  <main>
    <h1>DRPO Remasking Trace Demo</h1>
    <p class="subtitle">Parallel examples from real tokenizer-block traces. Pick a sample to inspect decode blocks and remask phases.</p>
    <div class="tabs" data-tabs></div>
    <section data-cases></section>
  </main>
  <script>
    const cases = {data_json};
    const tabs = document.querySelector("[data-tabs]");
    const caseRoot = document.querySelector("[data-cases]");
    const states = new Map();

    function makeButton(text, className = "control-button") {{
      const button = document.createElement("button");
      button.type = "button";
      button.className = className;
      button.textContent = text;
      return button;
    }}

    function renderDots(card, steps, activeIndex) {{
      const dots = card.querySelector("[data-dots]");
      dots.textContent = "";
      steps.forEach((_, index) => {{
        const dot = document.createElement("span");
        dot.className = index === activeIndex ? "dot active" : "dot";
        dot.dataset.index = String(index);
        dots.appendChild(dot);
      }});
    }}

    function renderStep(card, caseData) {{
      const index = states.get(caseData.id) ?? 0;
      const step = caseData.steps[index];
      card.querySelector("[data-step-title]").textContent = step.title;
      const focus = card.querySelector("[data-focus-line]");
      focus.innerHTML = step.focus;
      const viewport = card.querySelector(".viewport");
      viewport.scrollTop = viewport.scrollHeight;
      renderDots(card, caseData.steps, index);
      card.querySelector("[data-prev-step]").disabled = index === 0;
      card.querySelector("[data-next-step]").textContent = index === caseData.steps.length - 1 ? "Restart" : "Next step";
    }}

    function activateCase(id) {{
      document.querySelectorAll(".tab").forEach((tab) => tab.classList.toggle("active", tab.dataset.caseId === id));
      document.querySelectorAll(".case-card").forEach((card) => card.classList.toggle("active", card.dataset.caseId === id));
    }}

    cases.forEach((caseData, caseIndex) => {{
      states.set(caseData.id, 0);

      const tab = makeButton(caseData.id, "tab");
      tab.dataset.caseId = caseData.id;
      tab.addEventListener("click", () => activateCase(caseData.id));
      tabs.appendChild(tab);

      const card = document.createElement("article");
      card.className = "card case-card";
      card.dataset.caseId = caseData.id;
      card.innerHTML = `
        <div class="meta">
          <span class="pill">${{caseData.id}}</span>
          <span class="pill">${{caseData.correct ? "correct" : "incorrect"}}</span>
          <span class="pill">${{caseData.change}}</span>
          <span class="pill">block ${{caseData.block}}</span>
          <span class="pill">${{caseData.steps.length}} steps</span>
        </div>
        <p class="prompt">${{caseData.prompt}}</p>
        <div class="player">
          <div class="player-status">
            <p class="step-title" data-step-title></p>
            <div class="controls">
              <button type="button" class="control-button secondary" data-prev-step>Back</button>
              <button type="button" class="control-button" data-next-step>Next step</button>
            </div>
          </div>
          <div class="viewport" aria-live="polite">
            <p class="focus-line" data-focus-line></p>
          </div>
          <div class="dots" data-dots aria-hidden="true"></div>
        </div>
        <p class="note">Black frame = exact tokenizer block or remasked token from trace. Yellow background = remask window.</p>
      `;
      caseRoot.appendChild(card);

      card.querySelector("[data-prev-step]").addEventListener("click", () => {{
        states.set(caseData.id, Math.max(0, (states.get(caseData.id) ?? 0) - 1));
        renderStep(card, caseData);
      }});
      card.querySelector("[data-next-step]").addEventListener("click", () => {{
        const index = states.get(caseData.id) ?? 0;
        states.set(caseData.id, index === caseData.steps.length - 1 ? 0 : index + 1);
        renderStep(card, caseData);
      }});
      card.querySelector("[data-dots]").addEventListener("click", (event) => {{
        if (!event.target.matches(".dot")) return;
        states.set(caseData.id, Number(event.target.dataset.index));
        renderStep(card, caseData);
      }});
      renderStep(card, caseData);

      if (caseIndex === 0) activateCase(caseData.id);
    }});
  </script>
</body>
</html>
"""


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    cases = load_cases()
    records_by_prompt = load_records_by_prompt(cases)

    payloads = []
    for case in cases:
        records = records_by_prompt.get(case["example_abbr"])
        if not records:
            raise RuntimeError(f"No event records for {case['example_abbr']}")
        steps = build_steps(tokenizer, records)
        payloads.append(
            {
                "id": case["example_abbr"],
                "correct": bool(case["correct"]),
                "block": case["block_idx"],
                "change": f"{case['before_span']} → {case['after_span']}",
                "prompt": short_prompt(case),
                "steps": steps,
            }
        )

    html_text = build_html(payloads)
    TARGET_HTML.write_text(html_text, encoding="utf-8")
    LOCAL_COPY.write_text(html_text, encoding="utf-8")
    print(f"wrote {TARGET_HTML}")
    print(f"wrote {LOCAL_COPY}")
    for payload in payloads:
        print(f"{payload['id']}: {len(payload['steps'])} steps")


if __name__ == "__main__":
    main()
