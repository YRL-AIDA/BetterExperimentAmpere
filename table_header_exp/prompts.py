from typing import Dict, List

from .config import Config, MAX_TOKENS_BY_PROMPT


def is_thinking_prompt(cfg: Config, prompt_name: str) -> bool:
    return any(prompt_name.startswith(p) for p in cfg.thinking_prompt_prefixes)


def prepare_messages(system_prompt: str, prompt_config: Dict, table_repr: str,
                     table_format: str, chunk_info: str = "",
                     system_suffix: str = "") -> List[Dict[str, str]]:
    tpl = str(prompt_config.get("user", ""))
    up = (tpl
          .replace("{table_json}", table_repr)
          .replace("{table_html}", table_repr)
          .replace("{table_text}", table_repr)
          .replace("{table}", table_repr))
    if up == tpl:
        label = "HTML TABLE" if table_format == "html" else "TABLE (JSON)"
        up = f"{tpl}\n\n{label}:\n{table_repr}"
    if chunk_info:
        up += f"\n\n[NOTE: {chunk_info}]"
    sys = system_prompt
    if system_suffix:
        sys = f"{system_prompt}\n\n{system_suffix}"
    return [
        {"role": "system", "content": sys},
        {"role": "user", "content": up},
    ]


def get_requested_max_tokens(cfg: Config, prompt_name: str,
                             table_rows: int = 0, table_cols: int = 0,
                             thinking: bool = True) -> int:
    ceiling = cfg.context_window - cfg.context_safety_margin
    if not thinking:
        return max(cfg.min_completion_tokens, min(cfg.max_tokens_nonthinking, ceiling))
    base = MAX_TOKENS_BY_PROMPT.get(prompt_name, cfg.max_tokens)
    if table_rows > 0 or table_cols > 0:
        estimated_headers = table_cols * 2 + int(table_rows * 0.3)
        dynamic = base + estimated_headers * 15
        return max(cfg.min_completion_tokens, min(dynamic, ceiling))
    return max(cfg.min_completion_tokens, min(base, ceiling))
