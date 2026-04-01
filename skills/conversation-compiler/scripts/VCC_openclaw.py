#!/usr/bin/env python3
"""OpenClaw lexer/parser for VCC.

Transforms OpenClaw JSONL session logs into the same IR that VCC uses,
allowing all downstream passes (line assignment, brief/view lowering, emit)
to work unchanged.

OpenClaw JSONL record types:
  session            – session metadata (cwd, version, id)
  model_change       – model switch
  thinking_level_change – thinking budget change
  custom             – model-snapshot, etc.
  message            – core content: user, assistant, toolResult
  compaction         – context window compaction boundary

Message content block types:
  text      – plain text (user/assistant)
  toolCall  – tool invocation (name, id, arguments)
  thinking  – chain-of-thought (with signature)

Usage: imported by VCC.py or run standalone:
  python VCC_openclaw.py session.jsonl
  python VCC_openclaw.py session.jsonl --grep "pattern"
  python VCC_openclaw.py ~/.openclaw/agents/main/sessions/*.jsonl --grep "error"
"""

import json
import os
import re
import sys

# Import shared infrastructure from VCC
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import VCC
from VCC import (
    SEP, _node, _sanitize, _yaml_dump, _short_tid, _short,
    _preprocess_tool_text, _tool_summary, _collect_stats,
    assign_lines, lower_brief, lower_view, emit, match_lines,
    grep_search, _expand_inputs, _tokenize,
)

# ── OpenClaw-specific constants ──

# Record types to discard entirely (no conversational content)
_OC_DISCARD_TYPES = {
    "session", "model_change", "thinking_level_change", "custom",
}

# Tool names whose calls are internal bookkeeping (hide in brief mode)
_OC_INTERNAL_TOOLS = {"memory_search", "memory_get", "session_status"}

# ── OpenClaw tool summary ──

_OC_TOOL_SUMMARY_FIELDS = {
    "read": "path",
    "write": "path",
    "edit": "path",
    "exec": "command",
    "process": "action",
    "web_search": "query",
    "web_fetch": "url",
    "cron": "action",
    "image_generate": "prompt",
    "sessions_spawn": "task",
    "sessions_send": "message",
    "memory_search": "query",
    "memory_get": "path",
}

def _oc_tool_summary(name, args):
    """Build one-line summary for OpenClaw tools."""
    field = _OC_TOOL_SUMMARY_FIELDS.get(name)
    if field and field in args:
        val = str(args[field])
        if len(val) > 80:
            val = val[:77] + "..."
        return f'* {name} "{val}"'
    # Fallback: show first arg value
    if args:
        first_key = next(iter(args))
        val = str(args[first_key])
        if len(val) > 60:
            val = val[:57] + "..."
        return f'* {name} {first_key}="{val}"'
    return f"* {name}"


# ── OpenClaw lexer ──

def oc_lex(path):
    """Read OpenClaw JSONL, return list of raw records."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


# ── OpenClaw chain splitting ──

def oc_split_chains(recs):
    """Split records at compaction boundaries.

    OpenClaw compaction records mark where the context window was compressed.
    Each chain is one continuous conversation segment.
    """
    chains, cur = [], []
    for r in recs:
        if r.get("type") == "compaction":
            if cur:
                chains.append(cur)
            # Start new chain with compaction summary as context
            cur = [r]
        elif r.get("type") not in _OC_DISCARD_TYPES:
            cur.append(r)
    if cur:
        chains.append(cur)
    return chains if chains else [[]]


# ── OpenClaw message merging ──

def oc_merge_chunks(recs):
    """Merge consecutive assistant messages that share the same responseId.

    OpenClaw may split an assistant response across multiple JSONL records
    when streaming or when context compaction occurs mid-response.
    """
    merged = []
    active_rid = None
    active_idx = None

    for r in recs:
        if r.get("type") != "message":
            merged.append(r)
            active_rid = None
            active_idx = None
            continue

        msg = r.get("message", {})
        role = msg.get("role", "")

        if role == "assistant":
            rid = msg.get("responseId")
            if rid and rid == active_rid and active_idx is not None:
                # Merge content blocks
                merged[active_idx]["message"]["content"].extend(
                    msg.get("content", []))
                # Update usage if present
                if msg.get("usage"):
                    merged[active_idx]["message"]["usage"] = msg["usage"]
            else:
                merged.append(r)
                active_rid = rid
                active_idx = len(merged) - 1
        else:
            merged.append(r)
            active_rid = None
            active_idx = None

    return merged


# ── OpenClaw parser ──

def oc_parse(chain, outdir, data_prefix, data_ctr):
    """Parse OpenClaw records into VCC IR nodes."""
    ir = []
    sec = 0
    blk = 0

    # Build toolCallId → toolName mapping
    tid_name = {}
    for r in chain:
        if r.get("type") != "message":
            continue
        msg = r.get("message", {})
        if isinstance(msg.get("content"), list):
            for b in msg["content"]:
                if b.get("type") == "toolCall":
                    tid_name[b.get("id", "")] = b.get("name", "unknown")

    def _emit_sep():
        if sec > 0:
            ir.append(_node("meta", ["", SEP]))

    def _emit_header(h):
        ir.append(_node("meta_header", [h, ""], _sec=sec))

    # Process compaction record (if chain starts with one)
    for r in chain:
        if r.get("type") == "compaction":
            summary = r.get("summary", "")
            tokens_before = r.get("tokensBefore", 0)
            _emit_sep()
            _emit_header("[compaction]")
            header_line = f"[context compacted — {tokens_before} tokens before]"
            ir.append(_node("system", [header_line], searchable=False,
                           _sec=sec, _blk=blk))
            blk += 1
            if summary.strip():
                ir.append(_node("system", _sanitize(summary).split("\n"),
                               searchable=True, _sec=sec, _blk=blk))
                blk += 1
            sec += 1

    # Process message records
    for r in chain:
        if r.get("type") != "message":
            continue

        msg = r.get("message", {})
        role = msg.get("role", "")
        content = msg.get("content", [])

        if role == "user":
            # User message
            if isinstance(content, str):
                text = _sanitize(content)
                if text.strip():
                    _emit_sep()
                    _emit_header("[user]")
                    ir.append(_node("user", text.split("\n"),
                                   searchable=True, _sec=sec, _blk=blk))
                    blk += 1
                    sec += 1
            elif isinstance(content, list):
                texts = []
                for b in content:
                    if b.get("type") == "text":
                        t = _sanitize(b.get("text", ""))
                        if t.strip():
                            texts.append(t)
                if texts:
                    _emit_sep()
                    _emit_header("[user]")
                    full_text = "\n\n".join(texts)
                    ir.append(_node("user", full_text.split("\n"),
                                   searchable=True, _sec=sec, _blk=blk))
                    blk += 1
                    sec += 1

        elif role == "assistant":
            # Assistant message with mixed content blocks
            blocks = content if isinstance(content, list) else []
            has_content = any(
                (b.get("type") == "thinking" and b.get("thinking")) or
                (b.get("type") == "text" and b.get("text")) or
                b.get("type") == "toolCall"
                for b in blocks
            )
            if not has_content:
                continue

            _emit_sep()
            _emit_header("[assistant]")

            for b in blocks:
                bt = b.get("type")

                if bt == "thinking":
                    txt = _sanitize(b.get("thinking", ""))
                    if not txt:
                        continue
                    ir.append(_node("meta", [">>>thinking"],
                                   _sec=sec, _blk=blk))
                    ir.append(_node("thinking", txt.split("\n"),
                                   searchable=True, _sec=sec, _blk=blk))
                    ir.append(_node("meta", ["<<<thinking"],
                                   _sec=sec, _blk=blk))
                    blk += 1

                elif bt == "text":
                    txt = _sanitize(b.get("text", ""))
                    if not txt:
                        continue
                    ir.append(_node("assistant", txt.split("\n"),
                                   searchable=True, _sec=sec, _blk=blk))
                    blk += 1

                elif bt == "toolCall":
                    name = b.get("name", "unknown")
                    tid = b.get("id", "")
                    args = b.get("arguments", {})

                    hl = f">>>tool_call {name}:{_short_tid(tid)}"
                    summary = _oc_tool_summary(name, args)

                    ir.append(_node("meta", [hl], _sec=sec, _blk=blk,
                                   _tool_summary=summary))
                    if args:
                        ir.append(_node("tool_call",
                                       _yaml_dump(args).split("\n"),
                                       searchable=True, _sec=sec, _blk=blk))
                    ir.append(_node("meta", ["<<<tool_call"],
                                   _sec=sec, _blk=blk))
                    blk += 1

            sec += 1

        elif role == "toolResult":
            # Tool result
            tool_call_id = msg.get("toolCallId", "")
            tool_name = msg.get("toolName", "unknown")
            is_error = msg.get("isError", False)

            role_tag = "tool_error" if is_error else "tool"
            btype = "tool_error" if is_error else "tool_result"

            _emit_sep()
            _emit_header(f"[{role_tag}] {tool_name}:{_short_tid(tool_call_id)}")

            parts = []
            if isinstance(content, str):
                parts.append(_preprocess_tool_text(content, tool_name))
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict):
                        if item.get("type") == "text":
                            parts.append(_preprocess_tool_text(
                                item.get("text", ""), tool_name))
                    elif isinstance(item, str):
                        parts.append(item)

            text = _sanitize("\n\n".join(parts))
            if text.strip():
                ir.append(_node(btype, text.split("\n"),
                               searchable=True, _sec=sec, _blk=blk))
            else:
                ir.append(_node(btype, ["(empty)"],
                               searchable=False, _sec=sec, _blk=blk))
            blk += 1
            sec += 1

    ir.append(_node("meta", [""]))  # trailing newline
    return ir


# ── OpenClaw stats collection ──

def oc_collect_stats(chain):
    """Extract usage/timing/model stats from OpenClaw records."""
    from collections import defaultdict
    from datetime import datetime

    totals = defaultdict(lambda: 0.0)
    models = set()
    timestamps = []
    api_calls = 0
    tool_uses = 0

    for r in chain:
        ts = r.get("timestamp")
        if ts:
            timestamps.append(ts)

        if r.get("type") != "message":
            continue

        msg = r.get("message", {})
        model = msg.get("model")
        if model:
            models.add(model)

        usage = msg.get("usage")
        if usage:
            api_calls += 1
            totals["input"] += usage.get("input", 0)
            totals["output"] += usage.get("output", 0)
            totals["cacheRead"] += usage.get("cacheRead", 0)
            totals["cacheWrite"] += usage.get("cacheWrite", 0)
            totals["totalTokens"] += usage.get("totalTokens", 0)
            cost = usage.get("cost", {})
            if isinstance(cost, dict):
                totals["cost"] += cost.get("total", 0)

        # Count tool calls
        if isinstance(msg.get("content"), list):
            for b in msg["content"]:
                if isinstance(b, dict) and b.get("type") == "toolCall":
                    tool_uses += 1

    if api_calls == 0:
        return None

    duration = None
    if len(timestamps) >= 2:
        try:
            ts_list = sorted(timestamps)
            t0 = datetime.fromisoformat(ts_list[0].replace("Z", "+00:00"))
            t1 = datetime.fromisoformat(ts_list[-1].replace("Z", "+00:00"))
            duration = int((t1 - t0).total_seconds())
        except Exception:
            pass

    lines = [SEP, "[stats]", ""]
    if models:
        # Shorten model names for readability
        short_models = []
        for m in sorted(models):
            parts = m.rsplit("/", 1)
            short_models.append(parts[-1] if len(parts) > 1 else m)
        lines.append(f"model: {', '.join(short_models)}")

    lines.append(f"api_calls: {api_calls}  tool_uses: {tool_uses}")

    if duration is not None:
        m, s = divmod(duration, 60)
        lines.append(f"duration: {m}m{s:02d}s" if m else f"duration: {s}s")

    inp = int(totals["input"])
    cr = int(totals["cacheRead"])
    cw = int(totals["cacheWrite"])
    out = int(totals["output"])
    total = int(totals["totalTokens"])
    cost = totals["cost"]

    parts = []
    if inp:
        parts.append(f"input: {inp:,}")
    if cr:
        parts.append(f"cache_read: {cr:,}")
    if cw:
        parts.append(f"cache_write: {cw:,}")
    if parts:
        lines.append("  ".join(parts))
    lines.append(f"output: {out:,}")
    lines.append(f"total_tokens: {total:,}")
    if cost > 0:
        lines.append(f"cost: ${cost:.4f}")

    return lines


# ── OpenClaw brief-mode overrides ──

# OpenClaw injects cron prompts and system directives as user messages.
# These should be hidden in brief mode similarly to Claude Code's harness markup.
_OC_CRON_RE = re.compile(r'^\[cron:[0-9a-f-]+\s+')
_OC_HEARTBEAT_RE = re.compile(
    r'^(Read HEARTBEAT\.md|HEARTBEAT_OK|Heartbeat prompt:)', re.MULTILINE)


def oc_lower_brief(ir, truncate, filename="", truncate_user=256):
    """OpenClaw-specific brief lowering.

    Delegates to VCC's lower_brief, then applies OpenClaw-specific
    adjustments (e.g., hiding internal tools).
    """
    # Use the standard brief lowering
    lower_brief(ir, truncate, filename, truncate_user)

    # Post-process: hide OpenClaw internal tools in brief mode
    for o in ir:
        if o.get("content_brief") is None:
            continue
        if o["type"] == "meta" and o.get("content", []):
            c0 = o["content"][0]
            if c0.startswith(">>>tool_call"):
                tool_name = c0.split()[1].split(":")[0] if len(c0.split()) > 1 else ""
                if tool_name in _OC_INTERNAL_TOOLS:
                    o["content_brief"] = None


# ── compile (OpenClaw variant) ──

def oc_compile_pass(input_path, output_dir=None, truncate=128, truncate_user=256,
                    grep_pattern=None, quiet=False):
    """Compile an OpenClaw JSONL session into VCC views."""
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(input_path)) or "."
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(input_path))[0]

    recs = oc_merge_chunks(oc_lex(input_path))
    chains = oc_split_chains(recs)
    if not chains or not chains[0]:
        if not quiet:
            print("No conversation chains found.")
        return []

    results, paths = [], []

    for i, chain in enumerate(chains):
        sfx = f"_{i+1}" if len(chains) > 1 else ""
        ffn = f"{base}{sfx}.txt"
        mfn = f"{base}{sfx}.min.txt"
        vfn = f"{base}{sfx}.view.txt"
        fp = os.path.join(output_dir, ffn)
        mp = os.path.join(output_dir, mfn)
        vp = os.path.join(output_dir, vfn)
        data_ctr = [0]

        ir = oc_parse(chain, output_dir, f"{base}{sfx}", data_ctr)
        assign_lines(ir)
        oc_lower_brief(ir, truncate, ffn, truncate_user)

        full = emit(ir, "content")
        brief = emit(ir, "content_brief")

        stats_footer = oc_collect_stats(chain)
        if stats_footer:
            full.extend([""] + stats_footer)

        with open(fp, "w", encoding="utf-8") as f:
            f.write("\n".join(full))
        with open(mp, "w", encoding="utf-8") as f:
            f.write("\n".join(brief))

        if grep_pattern:
            lower_view(ir, ffn, grep_pattern)
            view = emit(ir, "content_view")
            with open(vp, "w", encoding="utf-8") as f:
                f.write("\n".join(view))

        ft, bt = "\n".join(full), "\n".join(brief)
        _cnt = lambda s: sum(1 for t in _tokenize(s) if t.strip())
        results.append((fp, ir))
        paths.append((fp, mp, vp if grep_pattern else None,
                       len(full), _cnt(ft), len(brief), _cnt(bt)))

    if not quiet:
        for fp, _, _, fl, fw, _, _ in paths:
            print(f"  {fp}  ({fl} lines, {fw} words)")
        for _, mp, _, _, _, bl, bw in paths:
            print(f"  {mp}  ({bl} lines, {bw} words)")
        if grep_pattern:
            for _, _, vp, _, _, _, _ in paths:
                if vp:
                    print(f"  {vp}")

    return results


# ── format auto-detection ──

def detect_format(path):
    """Detect whether a JSONL file is Claude Code or OpenClaw format.

    Returns 'openclaw', 'claude', or 'unknown'.
    """
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                t = obj.get("type", "")

                # OpenClaw: first record is typically "session" with version field
                if t == "session" and "version" in obj:
                    return "openclaw"

                # OpenClaw: model_change with provider field
                if t == "model_change" and "provider" in obj:
                    return "openclaw"

                # OpenClaw: message records have message.role
                if t == "message" and isinstance(obj.get("message"), dict):
                    msg = obj["message"]
                    if "role" in msg and isinstance(msg.get("content"), list):
                        for b in msg["content"]:
                            if isinstance(b, dict):
                                # OpenClaw uses "toolCall", Claude uses "tool_use"
                                if b.get("type") == "toolCall":
                                    return "openclaw"
                                if b.get("type") == "tool_use":
                                    return "claude"
                    return "openclaw"  # message with role field

                # Claude Code: uses "user"/"assistant"/"system" as top-level type
                if t in ("user", "assistant", "system"):
                    return "claude"

                # Claude Code: queue-operation, file-history-snapshot etc.
                if t in ("queue-operation", "file-history-snapshot",
                         "last-prompt", "progress"):
                    return "claude"
    except Exception:
        pass
    return "unknown"


# ── main (standalone) ──

def main():
    import argparse
    import io

    p = argparse.ArgumentParser(
        description="VCC OpenClaw - View-oriented Conversation Compiler for OpenClaw")
    p.add_argument("input", nargs="+",
                   help="OpenClaw JSONL session files (supports glob)")
    p.add_argument("-o", "--output-dir",
                   help="Output directory (default: same as input)")
    p.add_argument("-t", "--truncate", nargs="?", type=int,
                   const=128, default=128, metavar="N",
                   help="Token truncation limit for tool results (default: 128)")
    p.add_argument("-tu", "--truncate-user", nargs="?", type=int,
                   const=256, default=256, metavar="N",
                   help="Token truncation limit for user messages (default: 256)")
    p.add_argument("--grep", metavar="PATTERN",
                   help="Search pattern (regex) for adaptive view")
    p.add_argument("--auto", action="store_true",
                   help="Auto-detect format (OpenClaw vs Claude Code)")
    a = p.parse_args()

    try:
        grep_re = re.compile(a.grep, re.IGNORECASE) if a.grep else None
    except re.error as e:
        p.error(f"invalid regex for --grep: {e}")

    if sys.stdout.encoding and sys.stdout.encoding.lower().replace("-", "") != "utf8":
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", errors="replace")

    all_results = []
    for f in _expand_inputs(a.input):
        if a.auto:
            fmt = detect_format(f)
            if fmt == "claude":
                res = VCC.compile_pass(f, a.output_dir, a.truncate,
                                       a.truncate_user, grep_re,
                                       quiet=bool(grep_re))
            else:
                res = oc_compile_pass(f, a.output_dir, a.truncate,
                                      a.truncate_user, grep_re,
                                      quiet=bool(grep_re))
        else:
            res = oc_compile_pass(f, a.output_dir, a.truncate,
                                  a.truncate_user, grep_re,
                                  quiet=bool(grep_re))
        all_results.extend(res)

    if grep_re:
        grep_search(all_results, grep_re)


if __name__ == "__main__":
    main()
