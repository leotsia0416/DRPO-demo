#!/usr/bin/env python3
import html
import json
from pathlib import Path

from transformers import AutoTokenizer


REPO_ROOT = Path("/work/leotsia0416/projects/SDAR")
DEMO_ROOT = Path("/work/leotsia0416/DRPO-demo")
MODEL_PATH = REPO_ROOT / "checkpoint/grpo_from_remask_head_ref_only_90861/checkpoint-450"
EVENT_TRACE = (
    REPO_ROOT
    / "outputs/math500_grpo90861_ckpt450_850_950_rt050_bs16_8gpu_20260614_160254/checkpoint-450/rt0_50/remask_event_trace.jsonl"
)
SOURCE_HTML = DEMO_ROOT / "index.html"
TARGET_HTML = DEMO_ROOT / "index.html"
LOCAL_COPY = REPO_ROOT / "outputs/remask_visual_cases/key_sentence_inline_edits.html"
PROMPT_NEEDLE = "Roslyn has ten boxes"


def decode(tokenizer, token_ids):
    return tokenizer.decode(token_ids, skip_special_tokens=False)


def esc(text):
    return html.escape(text, quote=False).replace("&lt;|MASK|&gt;", '<span class="mask">MASK</span>')


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
    local_positions = [pos - prompt_length for pos in positions]
    local_pos = local_positions[0]

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


def load_records():
    records = []
    with EVENT_TRACE.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            prompt = (record.get("input") or [{}])[0].get("prompt", "")
            if PROMPT_NEEDLE in prompt:
                records.append(record)
    if not records:
        raise RuntimeError(f"No records found for prompt containing {PROMPT_NEEDLE!r}")
    return records


def build_steps(tokenizer, records):
    steps = [{"title": "Step 00 / Empty answer", "focus": '<span class="empty-token">0 generated tokens</span>'}]
    prev_after = []
    visible_step = 1

    for record in records:
        before_ids = record["generated_before_token_ids"]
        after_ids = record["generated_after_token_ids"]

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

        if record.get("triggered"):
            positions = record.get("remasked_positions") or []
            steps.append(
                {
                    "title": f"Step {visible_step:02d} / block {record['block_idx']} remask phase 1: target",
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
                        "title": f"Step {visible_step:02d} / block {record['block_idx']} remask phase 2: MASK",
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

            if before_ids != after_ids:
                steps.append(
                    {
                        "title": f"Step {visible_step:02d} / block {record['block_idx']} remask phase 3: refill",
                        "focus": render_old_new_window(tokenizer, before_ids, after_ids, record),
                    }
                )
                visible_step += 1

            steps.append(
                {
                    "title": f"Step {visible_step:02d} / block {record['block_idx']} remask phase 4: commit",
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

        prev_after = after_ids

    return steps


def replace_steps(html_text, steps):
    start = html_text.index("    const steps = ")
    end = html_text.index('    const player = document.querySelector("[data-remask-player]");')
    payload = json.dumps(steps, ensure_ascii=False, indent=6)
    block = (
        "    // Trace-driven steps generated from remask_event_trace.jsonl.\n"
        "    // Decode blocks highlight the exact tokenizer-id delta between adjacent records.\n"
        f"    const steps = {payload};\n\n"
    )
    return html_text[:start] + block + html_text[end:]


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    records = load_records()
    steps = build_steps(tokenizer, records)
    html_text = SOURCE_HTML.read_text(encoding="utf-8")
    html_text = replace_steps(html_text, steps)
    html_text = html_text.replace(
        "Sample math-500_404: click through generation from the first reasoning sentence to the remask window and final answer.",
        "Sample math-500_404: trace-driven playback from real tokenizer blocks and remask events.",
    )
    html_text = html_text.replace(
        '<span class="pill">window 316..328</span>',
        '<span class="pill">trace-driven</span><span class="pill">window 316..328</span>',
    )
    TARGET_HTML.write_text(html_text, encoding="utf-8")
    LOCAL_COPY.write_text(html_text, encoding="utf-8")
    print(f"wrote {TARGET_HTML}")
    print(f"wrote {LOCAL_COPY}")
    print(f"steps={len(steps)} records={len(records)}")


if __name__ == "__main__":
    main()
