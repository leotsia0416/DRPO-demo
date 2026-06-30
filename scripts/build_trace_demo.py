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
WINDOW_ONLY_EVENT_TRACE = (
    REPO_ROOT
    / "outputs/math500_ckpt450_rt050_window_only_nohard_start192_bs16_dev2_20260630_210844/remask_event_trace.jsonl"
)
REAL_PREDICTION_DIR = (
    REPO_ROOT
    / "outputs/math500_grpo90861_ckpt450_850_950_rt050_bs16_8gpu_20260614_160254/checkpoint-450/rt0_50/20260615_085443/predictions/checkpoint-450-gap-b4-thr0_95-rt0_50-t0_00-rs0_00-ri2-rw3-rstk192-pg192-tg1"
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

PATH_COMPARISON_CASE = "math-500_10"

SYNTHETIC_CHANGED_REMASKS_PER_CASE = 4
MAX_DISPLAYED_REMASKS_PER_CASE = 20
SYNTHETIC_REPLACEMENT_TEXTS_BY_KIND = {
    "word": [
        " pens",
        " pencils",
        " items",
        " terms",
        " values",
        " points",
        " lines",
        " number",
        " total",
        " sum",
        " difference",
        " ratio",
        " area",
    ],
    "capital_word": [
        "Total",
        "Number",
        "Sum",
        "Difference",
        "Ratio",
        "Area",
        "Thus",
        "Next",
    ],
    "number": [
        " 1",
        " 2",
        " 3",
        " 4",
        " 5",
        " 6",
        " 7",
        " 8",
        " 9",
    ],
    "operator": [
        " +",
        " -",
        " =",
        " \\times",
        " \\div",
    ],
    "punct": [
        ",",
        ".",
        ":",
        ";",
    ],
    "latex": [
        "$",
        "$$",
        "\\frac",
        "\\text",
    ],
}


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


def unpack_edit(edit_value):
    if isinstance(edit_value, tuple):
        if len(edit_value) == 3:
            return edit_value
        old_token_id, changed = edit_value
        return old_token_id, changed, None
    return edit_value, False, None


def render_inline_edit(tokenizer, edit_value, new_token_id):
    old_token_id, changed, change_number = unpack_edit(edit_value)
    marker_class = " edit-changed" if changed else ""
    body = (
        f'<span class="old{marker_class}">'
        + esc(decode(tokenizer, [old_token_id]))
        + "</span> "
        + f'<span class="new{marker_class}">'
        + esc(decode(tokenizer, [new_token_id]))
        + "</span>"
    )
    if changed and change_number is not None:
        return (
            f'<span class="change-pair" data-change="{change_number}">'
            + body
            + "</span>"
        )
    return body


def render_token_stream(
    tokenizer,
    token_ids,
    edits=None,
    decode_span=None,
    remask_window=None,
    highlight_positions=None,
    mask_old_tokens=None,
):
    edits = edits or {}
    highlight_positions = set(highlight_positions or [])
    mask_old_tokens = mask_old_tokens or {}
    remask_start, remask_end = remask_window or (None, None)
    decode_start, decode_end = decode_span or (None, None)

    parts = []
    window_open = False
    for pos, token_id in enumerate(token_ids):
        if remask_start is not None and pos == remask_start:
            parts.append('<span class="remask-window">')
            window_open = True

        if pos in mask_old_tokens:
            rendered = (
                '<span class="old">'
                + esc(decode(tokenizer, [mask_old_tokens[pos]]))
                + "</span> "
                + '<span class="mask">MASK</span>'
            )
        elif pos in edits:
            rendered = render_inline_edit(tokenizer, edits[pos], token_id)
        else:
            rendered = esc(decode(tokenizer, [token_id]))

        in_decode = (
            (decode_start is not None and decode_start <= pos < decode_end)
            or pos in highlight_positions
            or pos in mask_old_tokens
        )
        if in_decode:
            rendered = '<span class="decode-block">' + rendered + "</span>"
        parts.append(rendered)

        if remask_end is not None and pos + 1 == remask_end and window_open:
            parts.append("</span>")
            window_open = False

    if window_open:
        parts.append("</span>")
    return "".join(parts)


def render_compact_window(tokenizer, token_ids, highlight_index=None, old_new=None):
    parts = []
    for idx, token_id in enumerate(token_ids):
        if idx == highlight_index and old_new is not None:
            old_id, new_id = old_new
            rendered = render_inline_edit(tokenizer, (old_id, old_id != new_id, 1), new_id)
        else:
            rendered = esc(decode(tokenizer, [token_id]))
        if idx == highlight_index:
            rendered = '<span class="decode-block">' + rendered + "</span>"
        parts.append(rendered)
    return "".join(parts)


def record_local_window(record):
    start = max(0, record["window_token_start"] - record["prompt_length"])
    end = max(start, record["window_token_end"] - record["prompt_length"])
    return start, end


def record_local_positions(record):
    start, end = record_local_window(record)
    return [
        pos - record["prompt_length"]
        for pos in (record.get("remasked_positions") or [])
        if start <= pos - record["prompt_length"] < end
    ]


def record_local_positions_from(record, field_name):
    start, end = record_local_window(record)
    return [
        pos - record["prompt_length"]
        for pos in (record.get(field_name) or [])
        if start <= pos - record["prompt_length"] < end
    ]


def single_token_replacements(tokenizer):
    replacements = {}
    for kind, texts in SYNTHETIC_REPLACEMENT_TEXTS_BY_KIND.items():
        kind_ids = []
        for text in texts:
            token_ids = tokenizer.encode(text, add_special_tokens=False)
            if len(token_ids) == 1:
                kind_ids.append(token_ids[0])
        if kind_ids:
            replacements[kind] = kind_ids
    if not replacements:
        raise RuntimeError("No single-token synthetic replacements are available for this tokenizer.")
    return replacements


def replacement_kind_for_token(token_text):
    stripped = token_text.strip()
    if not stripped:
        return "punct"
    if stripped in {"+", "-", "=", "\\times", "\\div"}:
        return "operator"
    if stripped.replace(".", "", 1).isdigit():
        return "number"
    if stripped.startswith("\\") or stripped in {"$", "$$"}:
        return "latex"
    if all(not ch.isalnum() for ch in stripped):
        return "punct"
    if stripped[:1].isupper():
        return "capital_word"
    return "word"


def synthesize_changed_before_ids(tokenizer, before_ids, after_ids, record, replacement_ids, offset):
    positions = record.get("remasked_positions") or []
    if not positions:
        return before_ids

    local_pos = positions[0] - record["prompt_length"]
    if local_pos < 0 or local_pos >= min(len(before_ids), len(after_ids)):
        return before_ids

    refilled_id = after_ids[local_pos]
    refilled_text = decode(tokenizer, after_ids[local_pos : local_pos + 1])
    kind = replacement_kind_for_token(refilled_text)
    candidates = replacement_ids.get(kind) or replacement_ids.get("word") or next(iter(replacement_ids.values()))
    for step in range(len(candidates)):
        replacement_id = candidates[(offset + step) % len(candidates)]
        if replacement_id != refilled_id:
            patched = list(before_ids)
            patched[local_pos] = replacement_id
            return patched
    return before_ids


def dynamic_replacements_from_records(tokenizer, records, fallback_replacements):
    dynamic = {kind: [] for kind in fallback_replacements}
    seen = {kind: set() for kind in fallback_replacements}
    for record in records:
        if not record.get("triggered"):
            continue
        before_ids = record["generated_before_token_ids"]
        after_ids = record["generated_after_token_ids"]
        if before_ids != after_ids:
            continue
        local_pos = remasked_local_pos(record, before_ids, after_ids)
        if local_pos is None:
            continue
        token_id = after_ids[local_pos]
        token_text = decode(tokenizer, [token_id])
        if not is_readable_demo_token(token_text):
            continue
        kind = replacement_kind_for_token(token_text)
        if kind not in dynamic or token_id in seen[kind]:
            continue
        dynamic[kind].append(token_id)
        seen[kind].add(token_id)

    merged = {}
    for kind, fallback_ids in fallback_replacements.items():
        merged[kind] = dynamic.get(kind, []) + [token_id for token_id in fallback_ids if token_id not in seen.get(kind, set())]
    return merged


def remasked_local_pos(record, before_ids, after_ids):
    positions = record.get("remasked_positions") or []
    if not positions:
        return None
    local_pos = positions[0] - record["prompt_length"]
    if local_pos < 0 or local_pos >= min(len(before_ids), len(after_ids)):
        return None
    return local_pos


def is_readable_demo_token(token_text):
    stripped = token_text.strip()
    if not stripped:
        return False
    if stripped.startswith("<|") or stripped.endswith("|>"):
        return False
    return True


def select_synthetic_change_indices(tokenizer, records):
    candidates = []
    for idx, record in enumerate(records):
        if not record.get("triggered"):
            continue
        before_ids = record["generated_before_token_ids"]
        after_ids = record["generated_after_token_ids"]
        if before_ids != after_ids:
            continue
        local_pos = remasked_local_pos(record, before_ids, after_ids)
        if local_pos is None:
            continue
        token_text = decode(tokenizer, after_ids[local_pos : local_pos + 1])
        if not is_readable_demo_token(token_text):
            continue
        priority = 0 if token_text.strip().isalpha() else 1
        candidates.append((priority, idx))
    candidates.sort()
    return {idx for _, idx in candidates[:SYNTHETIC_CHANGED_REMASKS_PER_CASE]}


def select_displayed_remask_indices(records, synthetic_indices):
    selected = set(synthetic_indices)
    for idx, record in enumerate(records):
        if len(selected) >= MAX_DISPLAYED_REMASKS_PER_CASE:
            break
        if record.get("triggered"):
            selected.add(idx)
    return selected


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


def load_prompt_to_abbr():
    prompt_to_abbr = {}
    offset = 0
    prediction_files = sorted(
        REAL_PREDICTION_DIR.glob("math-500_*.json"),
        key=lambda path: int(path.stem.split("_")[-1]),
    )
    for prediction_file in prediction_files:
        payload = json.loads(prediction_file.read_text(encoding="utf-8"))
        for local_idx, item in payload.items():
            prompt_to_abbr[prompt_key(item["origin_prompt"])] = f"math-500_{offset + int(local_idx)}"
        offset += len(payload)
    return prompt_to_abbr


def load_window_only_would_events():
    events = defaultdict(list)
    if not WINDOW_ONLY_EVENT_TRACE.is_file():
        return events
    with WINDOW_ONLY_EVENT_TRACE.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if not record.get("would_trigger"):
                continue
            key = (
                prompt_key(record.get("input")),
                record.get("block_idx"),
                record.get("generated_blocks"),
                record.get("window_token_start"),
                record.get("window_token_end"),
            )
            events[key].append(record)
    return events


def load_event_records_for_abbr(event_path, target_abbr):
    prompt_to_abbr = load_prompt_to_abbr()
    records = []
    with event_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            abbr = prompt_to_abbr.get(prompt_key(record.get("input")))
            if abbr == target_abbr:
                records.append(record)
    records.sort(key=lambda item: (item.get("generated_blocks", -1), item.get("block_idx", -1)))
    return records


def selected_token_change(record):
    changes = []
    for position in record.get("remasked_positions") or []:
        offset = position - record["window_token_start"]
        before_ids = record.get("window_before_token_ids") or []
        after_ids = record.get("window_after_token_ids") or []
        if 0 <= offset < min(len(before_ids), len(after_ids)) and before_ids[offset] != after_ids[offset]:
            changes.append((position, offset, before_ids[offset], after_ids[offset]))
    return changes


def render_path_panel_step(
    tokenizer,
    token_ids,
    persistent_edits=None,
    decode_span=None,
    remask_window=None,
    highlight_positions=None,
    mask_old_tokens=None,
):
    if not token_ids:
        return '<span class="empty-token">0 generated tokens</span>'
    return render_token_stream(
        tokenizer,
        token_ids,
        edits=persistent_edits or {},
        decode_span=decode_span,
        remask_window=remask_window,
        highlight_positions=highlight_positions,
        mask_old_tokens=mask_old_tokens,
    )


def build_path_comparison_steps(tokenizer):
    real_records = load_event_records_for_abbr(EVENT_TRACE, PATH_COMPARISON_CASE)
    window_records = load_event_records_for_abbr(WINDOW_ONLY_EVENT_TRACE, PATH_COMPARISON_CASE)
    if not real_records or not window_records:
        return None

    window_by_generated = {record.get("generated_blocks"): record for record in window_records}
    prompt = (real_records[0].get("input") or [{}])[0].get("prompt", "")
    steps = [
        {
            "title": "Step 00 / empty answer",
            "window_title": "Window-only / no hard remask",
            "real_title": "Real remask",
            "window_focus": '<span class="empty-token">0 generated tokens</span>',
            "real_focus": '<span class="empty-token">0 generated tokens</span>',
        }
    ]

    prev_real_after = []
    prev_window_after = []
    real_persistent_edits = {}
    window_persistent_edits = {}
    changed_edit_count = 0
    visible_step = 1

    for real_record in real_records:
        window_record = window_by_generated.get(real_record.get("generated_blocks"))
        if not window_record:
            continue

        real_before = real_record.get("generated_before_token_ids") or []
        real_after = real_record.get("generated_after_token_ids") or real_before
        window_before = window_record.get("generated_before_token_ids") or []
        window_after = window_record.get("generated_after_token_ids") or window_before

        real_decode_span = None
        if real_before[: len(prev_real_after)] == prev_real_after and len(real_before) > len(prev_real_after):
            real_decode_span = (len(prev_real_after), len(real_before))

        window_decode_span = None
        if window_before[: len(prev_window_after)] == prev_window_after and len(window_before) > len(prev_window_after):
            window_decode_span = (len(prev_window_after), len(window_before))

        if real_decode_span or window_decode_span:
            steps.append(
                {
                    "title": (
                        f"Step {visible_step:02d} / block {real_record['block_idx']} synced decode"
                    ),
                    "window_title": f"window block {window_record['block_idx']}",
                    "real_title": f"real block {real_record['block_idx']}",
                    "window_focus": render_path_panel_step(
                        tokenizer,
                        window_before,
                        persistent_edits=window_persistent_edits,
                        decode_span=window_decode_span,
                    ),
                    "real_focus": render_path_panel_step(
                        tokenizer,
                        real_before,
                        persistent_edits=real_persistent_edits,
                        decode_span=real_decode_span,
                    ),
                }
            )
            visible_step += 1

        real_triggered = bool(real_record.get("triggered"))
        window_would_trigger = bool(window_record.get("would_trigger"))
        if real_triggered or window_would_trigger:
            real_window = record_local_window(real_record)
            window_window = record_local_window(window_record)
            real_positions = record_local_positions_from(real_record, "remasked_positions")
            window_positions = record_local_positions_from(window_record, "would_remasked_positions")

            real_masked_old_tokens = {
                local_pos: real_before[local_pos]
                for local_pos in real_positions
                if 0 <= local_pos < len(real_before)
            }
            real_current_edits = {}
            for local_pos in real_positions:
                if 0 <= local_pos < min(len(real_before), len(real_after)):
                    changed = real_before[local_pos] != real_after[local_pos]
                    change_number = None
                    if changed:
                        changed_edit_count += 1
                        change_number = changed_edit_count
                    real_current_edits[local_pos] = (real_before[local_pos], changed, change_number)

            steps.append(
                {
                    "title": f"Step {visible_step:02d} / block {real_record['block_idx']} remask target",
                    "window_title": "would target, hard remask suppressed",
                    "real_title": "target selected",
                    "window_focus": render_path_panel_step(
                        tokenizer,
                        window_before,
                        persistent_edits=window_persistent_edits,
                        remask_window=window_window,
                        highlight_positions=window_positions,
                    ),
                    "real_focus": render_path_panel_step(
                        tokenizer,
                        real_before,
                        persistent_edits=real_persistent_edits,
                        remask_window=real_window,
                        highlight_positions=real_positions,
                    ),
                }
            )
            visible_step += 1

            real_masked_ids = real_record.get("generated_with_masks_token_ids") or real_before
            steps.append(
                {
                    "title": f"Step {visible_step:02d} / block {real_record['block_idx']} mask/suppress",
                    "window_title": "suppressed: token remains visible",
                    "real_title": "mask applied",
                    "window_focus": render_path_panel_step(
                        tokenizer,
                        window_before,
                        persistent_edits=window_persistent_edits,
                        remask_window=window_window,
                        highlight_positions=window_positions,
                    ),
                    "real_focus": render_path_panel_step(
                        tokenizer,
                        real_masked_ids,
                        persistent_edits=real_persistent_edits,
                        remask_window=real_window,
                        highlight_positions=real_positions,
                        mask_old_tokens=real_masked_old_tokens,
                    ),
                }
            )
            visible_step += 1

            real_preview_edits = {**real_persistent_edits, **real_current_edits}
            steps.append(
                {
                    "title": f"Step {visible_step:02d} / block {real_record['block_idx']} refill/continue",
                    "window_title": "no refill: same token stream",
                    "real_title": "refilled token stream",
                    "window_focus": render_path_panel_step(
                        tokenizer,
                        window_after,
                        persistent_edits=window_persistent_edits,
                        remask_window=window_window,
                        highlight_positions=window_positions,
                    ),
                    "real_focus": render_path_panel_step(
                        tokenizer,
                        real_after,
                        persistent_edits=real_preview_edits,
                        remask_window=real_window,
                        highlight_positions=real_positions,
                    ),
                }
            )
            visible_step += 1

            real_persistent_edits.update(real_current_edits)
            steps.append(
                {
                    "title": f"Step {visible_step:02d} / block {real_record['block_idx']} commit",
                    "window_title": "window-only committed",
                    "real_title": "real remask committed",
                    "window_focus": render_path_panel_step(
                        tokenizer,
                        window_after,
                        persistent_edits=window_persistent_edits,
                        remask_window=window_window,
                        highlight_positions=window_positions,
                    ),
                    "real_focus": render_path_panel_step(
                        tokenizer,
                        real_after,
                        persistent_edits=real_persistent_edits,
                        remask_window=real_window,
                        highlight_positions=real_positions,
                    ),
                }
            )
            visible_step += 1

        prev_real_after = real_after
        prev_window_after = window_after

    strict_record = None
    strict_window_record = None
    for real_record in real_records:
        changes = selected_token_change(real_record)
        if not changes:
            continue
        window_record = window_by_generated.get(real_record.get("generated_blocks"))
        if not window_record:
            continue
        if real_record.get("window_before_token_ids") != window_record.get("window_before_token_ids"):
            continue
        if real_record.get("candidate_positions") != window_record.get("candidate_positions"):
            continue
        if real_record.get("remasked_positions") != window_record.get("would_remasked_positions"):
            continue
        strict_record = real_record
        strict_window_record = window_record
        break

    strict_change = None
    if strict_record is not None:
        position, offset, old_id, new_id = selected_token_change(strict_record)[0]
        strict_change = {
            "block": strict_record.get("block_idx"),
            "generated_blocks": strict_record.get("generated_blocks"),
            "position": position,
            "old": decode(tokenizer, [old_id]),
            "new": decode(tokenizer, [new_id]),
            "best_real": strict_record.get("best_score"),
            "best_window": strict_window_record.get("best_score") if strict_window_record else None,
        }

    return {
        "id": PATH_COMPARISON_CASE,
        "prompt": prompt[:360] + ("..." if len(prompt) > 360 else ""),
        "steps": steps,
        "strict_change": strict_change,
        "real_records": len(real_records),
        "window_records": len(window_records),
    }


def build_path_comparison(tokenizer):
    prompt_to_abbr = load_prompt_to_abbr()
    window_events = load_window_only_would_events()
    with EVENT_TRACE.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            real_record = json.loads(line)
            if not real_record.get("triggered"):
                continue
            abbr = prompt_to_abbr.get(prompt_key(real_record.get("input")))
            if abbr != PATH_COMPARISON_CASE:
                continue
            changes = selected_token_change(real_record)
            if not changes:
                continue
            key = (
                prompt_key(real_record.get("input")),
                real_record.get("block_idx"),
                real_record.get("generated_blocks"),
                real_record.get("window_token_start"),
                real_record.get("window_token_end"),
            )
            for window_record in window_events.get(key, []):
                if real_record.get("window_before_token_ids") != window_record.get("window_before_token_ids"):
                    continue
                if real_record.get("candidate_positions") != window_record.get("candidate_positions"):
                    continue
                if real_record.get("remasked_positions") != window_record.get("would_remasked_positions"):
                    continue
                position, offset, old_id, new_id = changes[0]
                prompt = (real_record.get("input") or [{}])[0].get("prompt", "")
                before_html = render_compact_window(
                    tokenizer,
                    real_record.get("window_before_token_ids") or [],
                    offset,
                )
                masked_html = render_compact_window(
                    tokenizer,
                    real_record.get("window_with_masks_token_ids") or [],
                    offset,
                )
                real_after_html = render_compact_window(
                    tokenizer,
                    real_record.get("window_after_token_ids") or [],
                    offset,
                    old_new=(old_id, new_id),
                )
                window_after_html = render_compact_window(
                    tokenizer,
                    window_record.get("window_after_token_ids") or [],
                    offset,
                )
                return {
                    "id": abbr,
                    "prompt": prompt[:360] + ("..." if len(prompt) > 360 else ""),
                    "block": real_record.get("block_idx"),
                    "generated_blocks": real_record.get("generated_blocks"),
                    "position": position,
                    "old": decode(tokenizer, [old_id]),
                    "new": decode(tokenizer, [new_id]),
                    "best_real": real_record.get("best_score"),
                    "best_window": window_record.get("best_score"),
                    "before": before_html,
                    "real_masked": masked_html,
                    "real_after": real_after_html,
                    "window_after": window_after_html,
                }
    return None


def build_steps(tokenizer, records):
    steps = [{"title": "Step 00 / Empty answer", "focus": '<span class="empty-token">0 generated tokens</span>'}]
    prev_after = []
    persistent_edits = {}
    visible_step = 1
    replacement_ids = dynamic_replacements_from_records(tokenizer, records, single_token_replacements(tokenizer))
    synthetic_indices = select_synthetic_change_indices(tokenizer, records)
    displayed_indices = select_displayed_remask_indices(records, synthetic_indices)
    synthetic_count = 0
    changed_edit_count = 0

    for record_idx, record in enumerate(records):
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
                    "focus": render_token_stream(
                        tokenizer,
                        before_ids,
                        edits=persistent_edits,
                        decode_span=(start, end),
                    ),
                }
            )
            visible_step += 1

        changed_by_remask = before_ids != after_ids
        synthetic_change = False
        if record_idx in synthetic_indices:
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

        if record.get("triggered") and record_idx in displayed_indices:
            local_window = record_local_window(record)
            local_positions = record_local_positions(record)
            masked_old_tokens = {
                local_pos: before_ids[local_pos]
                for local_pos in local_positions
                if 0 <= local_pos < len(before_ids)
            }
            current_edits = {}
            for local_pos in local_positions:
                if 0 <= local_pos < min(len(before_ids), len(after_ids)):
                    changed = before_ids[local_pos] != after_ids[local_pos]
                    change_number = None
                    if changed:
                        changed_edit_count += 1
                        change_number = changed_edit_count
                    current_edits[local_pos] = (before_ids[local_pos], changed, change_number)
            demo_note = " (demo token swap)" if synthetic_change else ""
            steps.append(
                {
                    "title": f"Step {visible_step:02d} / block {record['block_idx']} remask phase 1: target{demo_note}",
                    "focus": render_token_stream(
                        tokenizer,
                        before_ids,
                        edits=persistent_edits,
                        remask_window=local_window,
                        highlight_positions=local_positions,
                    ),
                }
            )
            visible_step += 1

            masked_ids = record.get("generated_with_masks_token_ids")
            if masked_ids:
                steps.append(
                    {
                        "title": f"Step {visible_step:02d} / block {record['block_idx']} remask phase 2: MASK{demo_note}",
                        "focus": render_token_stream(
                            tokenizer,
                            masked_ids,
                            edits=persistent_edits,
                            remask_window=local_window,
                            highlight_positions=local_positions,
                            mask_old_tokens=masked_old_tokens,
                        ),
                    }
                )
                visible_step += 1

            preview_edits = {**persistent_edits, **current_edits}
            steps.append(
                {
                    "title": f"Step {visible_step:02d} / block {record['block_idx']} remask phase 3: refill{demo_note}",
                    "focus": render_token_stream(
                        tokenizer,
                        after_ids,
                        edits=preview_edits,
                        remask_window=local_window,
                        highlight_positions=local_positions,
                    ),
                }
            )
            visible_step += 1

            persistent_edits.update(current_edits)
            steps.append(
                {
                    "title": f"Step {visible_step:02d} / block {record['block_idx']} remask phase 4: commit{demo_note}",
                    "focus": render_token_stream(
                        tokenizer,
                        after_ids,
                        edits=persistent_edits,
                        remask_window=local_window,
                        highlight_positions=local_positions,
                    ),
                }
            )
            visible_step += 1

        prev_after = original_after_ids

    return steps


def short_prompt(case):
    prompt = case["prompt"].replace(" Solve the problem step by step", "\nSolve the problem step by step")
    return prompt[:360] + ("..." if len(prompt) > 360 else "")


def build_html(case_payloads, comparison_payload):
    data_json = json.dumps(case_payloads, ensure_ascii=False)
    comparison_json = json.dumps(comparison_payload, ensure_ascii=False)
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
    .edit-changed {{
      outline: 2px solid #111;
      outline-offset: 1px;
      box-shadow: 0 0 0 3px rgba(255, 255, 255, 0.78);
    }}
    .new.edit-changed {{
      text-decoration: underline;
      text-decoration-thickness: 0.16em;
      text-underline-offset: 0.16em;
    }}
    .change-pair {{
      position: relative;
      display: inline-block;
      margin: 0 0.12em;
    }}
    .change-pair::before {{
      content: attr(data-change);
      position: absolute;
      left: -0.72em;
      top: -0.92em;
      min-width: 1.16em;
      height: 1.16em;
      border-radius: 999px;
      background: #2f6df6;
      color: white;
      border: 2px solid white;
      box-shadow: 0 1px 4px rgba(17, 17, 17, 0.24);
      font: 700 10px/1.02 ui-sans-serif, system-ui, sans-serif;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      z-index: 2;
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
    .comparison-card {{
      margin-top: 26px;
    }}
    .comparison-player-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 16px;
    }}
    .path-panel {{
      border: 1px solid var(--line);
      border-radius: 18px;
      background: rgba(255, 255, 255, 0.42);
      padding: 16px;
    }}
    .path-panel h3 {{
      margin: 0 0 10px;
      font: 700 16px/1.2 ui-sans-serif, system-ui, sans-serif;
    }}
    .mini-label {{
      margin: 12px 0 5px;
      color: var(--muted);
      font: 700 11px/1.2 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }}
    .mini-window {{
      border: 2px solid #171717;
      border-radius: 14px;
      background: rgba(255, 255, 255, 0.55);
      padding: 14px;
      min-height: 86px;
      font-size: 15px;
      line-height: 1.7;
      white-space: pre-wrap;
      overflow: auto;
    }}
    .path-panel .viewport {{
      min-height: 360px;
      max-height: 58vh;
      border-radius: 16px;
    }}
    .path-panel .viewport::before {{
      content: attr(data-panel-label);
    }}
    @media (max-width: 760px) {{
      .comparison-player-grid {{ grid-template-columns: 1fr; }}
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
    const comparison = {comparison_json};
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

    function scrollToActiveMark(viewport) {{
      const marks = viewport.querySelectorAll(".mask, .new, .old, .decode-block");
      const target = marks.length ? marks[marks.length - 1] : null;
      if (!target) {{
        viewport.scrollTop = viewport.scrollHeight;
        return;
      }}
      const viewportBox = viewport.getBoundingClientRect();
      const targetBox = target.getBoundingClientRect();
      const targetTop = viewport.scrollTop + targetBox.top - viewportBox.top;
      viewport.scrollTop = Math.max(0, targetTop - viewport.clientHeight * 0.38);
    }}

    function renderStep(card, caseData) {{
      const index = states.get(caseData.id) ?? 0;
      const step = caseData.steps[index];
      card.querySelector("[data-step-title]").textContent = step.title;
      const focus = card.querySelector("[data-focus-line]");
      focus.innerHTML = step.focus;
      const viewport = card.querySelector(".viewport");
      scrollToActiveMark(viewport);
      renderDots(card, caseData.steps, index);
      card.querySelector("[data-prev-step]").disabled = index === 0;
      card.querySelector("[data-next-step]").textContent = index === caseData.steps.length - 1 ? "Restart" : "Next step";
    }}

    function renderComparisonStep(card, comparisonData) {{
      const index = states.get(comparisonData.id) ?? 0;
      const step = comparisonData.steps[index];
      card.querySelector("[data-step-title]").textContent = step.title;
      card.querySelector("[data-window-title]").textContent = step.window_title;
      card.querySelector("[data-real-title]").textContent = step.real_title;
      const windowFocus = card.querySelector("[data-window-focus]");
      const realFocus = card.querySelector("[data-real-focus]");
      windowFocus.innerHTML = step.window_focus;
      realFocus.innerHTML = step.real_focus;
      card.querySelectorAll(".viewport").forEach(scrollToActiveMark);
      renderDots(card, comparisonData.steps, index);
      card.querySelector("[data-prev-step]").disabled = index === 0;
      card.querySelector("[data-next-step]").textContent = index === comparisonData.steps.length - 1 ? "Restart" : "Next step";
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
        <p class="note">Black frame = current decode or remask token. Yellow = remask window. Red strike = removed token; green = refill token, kept in later steps.</p>
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

    if (comparison) {{
      states.set(comparison.id, 0);

      const tab = makeButton(comparison.id, "tab");
      tab.dataset.caseId = comparison.id;
      tab.addEventListener("click", () => activateCase(comparison.id));
      tabs.appendChild(tab);

      const card = document.createElement("article");
      card.className = "card case-card comparison-card";
      card.dataset.caseId = comparison.id;
      const strict = comparison.strict_change;
      const strictMeta = strict
        ? `
          <span class="pill">block ${{strict.block}}</span>
          <span class="pill">token ${{strict.position}}</span>
          <span class="pill">${{strict.old}} → ${{strict.new}}</span>
        `
        : "";
      card.innerHTML = `
        <div class="meta">
          <span class="pill">synced path comparison</span>
          <span class="pill">${{comparison.id}}</span>
          <span class="pill">${{comparison.steps.length}} steps</span>
          <span class="pill">real records ${{comparison.real_records}}</span>
          <span class="pill">window records ${{comparison.window_records}}</span>
          ${{strictMeta}}
        </div>
        <p class="prompt">${{comparison.prompt}}</p>
        <div class="player">
          <div class="player-status">
            <p class="step-title" data-step-title></p>
            <div class="controls">
              <button type="button" class="control-button secondary" data-prev-step>Back</button>
              <button type="button" class="control-button" data-next-step>Next step</button>
            </div>
          </div>
          <div class="comparison-player-grid">
          <div class="path-panel">
              <h3 data-window-title></h3>
              <div class="viewport" aria-live="polite" data-panel-label="window-only">
                <p class="focus-line" data-window-focus></p>
              </div>
          </div>
          <div class="path-panel">
              <h3 data-real-title></h3>
              <div class="viewport" aria-live="polite" data-panel-label="real remask">
                <p class="focus-line" data-real-focus></p>
              </div>
            </div>
          </div>
          <div class="dots" data-dots aria-hidden="true"></div>
        </div>
        <p class="note">Left = window-only run with hard remask suppressed. Right = real remask run. The player advances both paths with the same step index.</p>
      `;
      caseRoot.appendChild(card);
      card.querySelector("[data-prev-step]").addEventListener("click", () => {{
        states.set(comparison.id, Math.max(0, (states.get(comparison.id) ?? 0) - 1));
        renderComparisonStep(card, comparison);
      }});
      card.querySelector("[data-next-step]").addEventListener("click", () => {{
        const index = states.get(comparison.id) ?? 0;
        states.set(comparison.id, index === comparison.steps.length - 1 ? 0 : index + 1);
        renderComparisonStep(card, comparison);
      }});
      card.querySelector("[data-dots]").addEventListener("click", (event) => {{
        if (!event.target.matches(".dot")) return;
        states.set(comparison.id, Number(event.target.dataset.index));
        renderComparisonStep(card, comparison);
      }});
      renderComparisonStep(card, comparison);
    }} else {{
      const card = document.createElement("article");
      card.className = "card";
      card.innerHTML = `<p class="note">Path comparison case is not available yet.</p>`;
      caseRoot.appendChild(card);
    }}
  </script>
</body>
</html>
"""


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    cases = load_cases()
    records_by_prompt = load_records_by_prompt(cases)
    comparison_payload = build_path_comparison_steps(tokenizer)

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

    html_text = build_html(payloads, comparison_payload)
    TARGET_HTML.write_text(html_text, encoding="utf-8")
    LOCAL_COPY.write_text(html_text, encoding="utf-8")
    print(f"wrote {TARGET_HTML}")
    print(f"wrote {LOCAL_COPY}")
    for payload in payloads:
        print(f"{payload['id']}: {len(payload['steps'])} steps")


if __name__ == "__main__":
    main()
