import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


EXCLUDE_DIR_NAMES = {
    "results", "raw_responses", "test_responses",
    "__pycache__", ".git", ".venv", "venv", "_logs",
}
EXCLUDE_FILE_PREFIXES = (
    "responses_", "failed_", "checkpoint_",
    "summary_", "parsed_responses_",
)
EXCLUDE_PROMPT_FILES = {"sc_max.txt", "sc_min.txt", "system.txt"}
LABEL_KEYS = {
    "is_column_header", "is_projected_row_header", "is_metadata",
    "is_row_header", "is_spanning", "label", "labels",
    "gold", "answer", "answers", "target",
}

MAX_TOKENS_BY_PROMPT: Dict[str, int] = {
    "reasoning_max":         24576,
    "reasoning_min":         24576,
    "reasoning_domain":      24576,
    "reasoning_few_domain":  24576,
    "fewshot_reasoning_max": 24576,
    "fewshot_reasoning_min": 24576,
}

FATAL_ERROR_TYPES = {"connection_error", "timeout", "oom", "api_error", "rate_limit"}


@dataclass
class Config:
    project_root: Path = field(default_factory=lambda: Path(os.getenv("PROJECT_ROOT", str(Path.cwd()))))
    vllm_base_url: str = os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8000/v1")
    vllm_api_key: str = os.getenv("VLLM_API_KEY", "EMPTY")
    model_name: str = os.getenv("MODEL_NAME", "Qwen/Qwen3.5-9B")
    model_alias: str = os.getenv("MODEL_ALIAS", "")

    output_dir: str = os.getenv("OUTPUT_DIR", "results")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")

    max_retries: int = int(os.getenv("MAX_RETRIES", "2"))
    retry_backoff_base: float = float(os.getenv("RETRY_BACKOFF_BASE", "3.0"))
    temperature: float = float(os.getenv("TEMPERATURE", "0.0"))
    seed: Optional[int] = (int(os.getenv("SEED")) if os.getenv("SEED") not in (None, "") else None)
    max_tokens: int = int(os.getenv("MAX_TOKENS", "16384"))

    concurrency: int = int(os.getenv("CONCURRENCY", "4"))
    checkpoint_every: int = int(os.getenv("CHECKPOINT_EVERY", "10"))
    snapshot_every: int = int(os.getenv("SNAPSHOT_EVERY", "500"))
    request_timeout_sec: float = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "0"))
    inter_request_delay: float = float(os.getenv("INTER_REQUEST_DELAY", "0"))
    early_stop_failures: int = int(os.getenv("EARLY_STOP_FAILURES", "10"))

    context_window: int = int(os.getenv("MODEL_CONTEXT_LIMIT", "32768"))
    context_safety_margin: int = int(os.getenv("CONTEXT_SAFETY_MARGIN", "512"))
    min_completion_tokens: int = int(os.getenv("MIN_COMPLETION_TOKENS", "256"))
    auto_detect_window: bool = os.getenv("AUTO_DETECT_WINDOW", "1") == "1"

    enable_continuation: bool = os.getenv("ENABLE_CONTINUATION", "1") == "1"
    disable_thinking_supported: bool = os.getenv("DISABLE_THINKING_SUPPORTED", "1") == "1"
    max_continuation_rounds: int = int(os.getenv("MAX_CONTINUATION_ROUNDS", "0"))
    force_answer_when_exhausted: bool = os.getenv("FORCE_ANSWER_WHEN_EXHAUSTED", "1") == "1"

    chunk_strategy: str = os.getenv("CHUNK_STRATEGY", "header_aware")
    header_zone_rows: int = int(os.getenv("HEADER_ZONE_ROWS", "6"))

    total_tables: int = int(os.getenv("TOTAL_TABLES", "0"))
    format_ratio: str = os.getenv("FORMAT_RATIO", "50:50")
    table_seed_path: Optional[str] = None

    def experiment_plan(self) -> List[Dict]:
        r = self.project_root
        return [
            {
                "name": "pubtables_complex_top500",
                "json_root": r / "Get_500_Tables_from_PubTables" / "JSON_Complex_TOP500_normalized",
                "html_root": r / "Get_500_Tables_from_PubTables" / "JSON_Complex_TOP500_normalized_html",
                "limit": 500,
                "prompts": ["zero_domain", "fewshot_domain", "reasoning_domain"],
            },
            {
                "name": "maximum_viewpoint",
                "json_root": r / "Convert_from_xlsx_to_Json" / "maximum_viewpoint_converted_json",
                "html_root": r / "Convert_from_json_to_html" / "maximum_viewpoint_converted_html",
                "limit": 500,
                "prompts": ["zero_max", "fewshot_max", "reasoning_max"],
            },
            {
                "name": "table_normalization",
                "json_root": r / "Convert_from_xlsx_to_Json" / "table_normalization_converted_json",
                "html_root": r / "Convert_from_json_to_html" / "table_normalization_converted_html",
                "limit": 500,
                "prompts": ["zero_min", "fewshot_min", "reasoning_min"],
            },
        ]

    @property
    def prompts_dir(self) -> Path:
        return self.project_root / "prompts"
