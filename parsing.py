import re
from typing import Any, Dict, List, Tuple


def parse_output(raw_text: str) -> Tuple[bool, List[Dict[str, Any]], str]:
    if not raw_text or not str(raw_text).strip():
        return True, [], ""

    text = str(raw_text).strip()

    think_close = "</think>"
    if think_close in text:
        text = text.split(think_close, 1)[1].strip()
    elif "<think>" in text:
        return False, [], "truncated_inside_think_block"

    text = re.sub(r"^```[a-z]*\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text, flags=re.IGNORECASE).strip()

    has_done = bool(re.search(r"^DONE\s*$", text, re.MULTILINE | re.IGNORECASE))
    text = re.sub(r"^DONE\s*$", "", text, flags=re.MULTILINE | re.IGNORECASE).strip()

    if not text:
        return True, [], ("done_marker" if has_done else "")

    seen = set()
    coords: List[Dict[str, Any]] = []

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if "|" in line:
            left, _, cell_text = line.partition("|")
            m = re.search(r"(\d+)\s+(\d+)", left)
            if m:
                r, c = int(m.group(1)), int(m.group(2))
                if (r, c) not in seen:
                    seen.add((r, c))
                    coords.append({"row": r, "col": c, "text": cell_text.strip()})
        else:
            m = re.fullmatch(r"(\d+)[\s,;]+(\d+)", line)
            if m:
                r, c = int(m.group(1)), int(m.group(2))
                if (r, c) not in seen:
                    seen.add((r, c))
                    coords.append({"row": r, "col": c, "text": ""})

    if coords:
        coords.sort(key=lambda x: (x["row"], x["col"]))
        return True, coords, ("done_marker" if has_done else "")

    pairs = re.findall(r"\[\s*(\d+)\s*,\s*(\d+)\s*\]", text)
    if pairs:
        for rs, cs in pairs:
            r, c = int(rs), int(cs)
            if (r, c) not in seen:
                seen.add((r, c))
                coords.append({"row": r, "col": c, "text": ""})
        coords.sort(key=lambda x: (x["row"], x["col"]))
        return True, coords, ("done_marker" if has_done else "fallback_json_array")

    parens = re.findall(r"\(\s*(\d+)\s*,\s*(\d+)\s*\)", text)
    if parens:
        for rs, cs in parens:
            r, c = int(rs), int(cs)
            if (r, c) not in seen:
                seen.add((r, c))
                coords.append({"row": r, "col": c, "text": ""})
        coords.sort(key=lambda x: (x["row"], x["col"]))
        return True, coords, ("done_marker" if has_done else "fallback_paren_format")

    return False, [], "no_parseable_coordinates"


def classify_api_error(msg: str) -> str:
    m = (msg or "").lower()
    if any(x in m for x in ["maximum context length", "context length", "too many tokens"]):
        return "context_length_exceeded"
    if "timeout" in m or "timed out" in m:
        return "timeout"
    if "out of memory" in m or "oom" in m or "cuda" in m:
        return "oom"
    if "all connection attempts failed" in m or "connection refused" in m:
        return "connection_error"
    if "rate limit" in m or "429" in m:
        return "rate_limit"
    if "bad request" in m or "400" in m:
        return "bad_request"
    return "api_error"


def parse_context_overflow(error_msg: str) -> Tuple:
    if "context length" not in error_msg and "maximum context" not in error_msg:
        return (None, None)
    prompt_toks = None
    for pat in (r"(\d+)\s+in the messages",
                r"prompt contains at least (\d+)\s+input tokens",
                r"(\d+)\s+input tokens"):
        m = re.search(pat, error_msg)
        if m:
            prompt_toks = int(m.group(1))
            break
    server_ctx = None
    m = re.search(r"maximum context length is (\d+)", error_msg)
    if m:
        server_ctx = int(m.group(1))
    return (prompt_toks, server_ctx)
