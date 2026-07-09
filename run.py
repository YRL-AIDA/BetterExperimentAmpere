import argparse
import logging
from pathlib import Path

from table_header_exp.config import Config
from table_header_exp.orchestrator import Collector


def main():
    parser = argparse.ArgumentParser(
        description="Table header detection experiment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--vllm-url", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--model-alias", default=None)
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--concurrency", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--timeout", type=float, default=None)
    parser.add_argument("--inter-delay", type=float, default=None)
    parser.add_argument("--early-stop", type=int, default=None)
    parser.add_argument("--total-tables", type=int, default=None)
    parser.add_argument("--format-ratio", default=None)
    parser.add_argument("--context-window", type=int, default=None)
    parser.add_argument("--no-auto-detect-window", action="store_true", default=False)
    parser.add_argument("--chunk-strategy", choices=["header_aware", "whole"], default=None)
    parser.add_argument("--header-zone-rows", type=int, default=None)
    parser.add_argument("--no-continuation", action="store_true", default=False)
    parser.add_argument("--max-continuation-rounds", type=int, default=None,
                        metavar="N", help="0 = unlimited (bounded by context budget)")
    parser.add_argument("--no-force-answer", action="store_true", default=False,
                        help="do not force a thinking-off answer as a last resort")
    parser.add_argument("--no-disable-thinking", action="store_true", default=False)
    parser.add_argument("--thinking-off-mode", default=None,
                        choices=["chat_template", "enable_thinking", "reasoning", "none"],
                        help="how to turn thinking off for non-reasoning prompts")
    parser.add_argument("--max-tokens-nonthinking", type=int, default=None)
    parser.add_argument("--no-tokenizer", action="store_true", default=False,
                        help="skip the /tokenize endpoint (hosted APIs lack it)")
    parser.add_argument("--cache-dir", default=None, metavar="DIR",
                        help="cache responses by request_id to avoid re-billing on reruns")
    parser.add_argument("--extra-body", default=None, metavar="JSON",
                        help='extra JSON merged into every request, e.g. '
                             '\'{"provider":{"allow_fallbacks":false,"country":"ru"}}\'')
    parser.add_argument("--table-seed", default=None, metavar="PATH")
    parser.add_argument("--log-level", default=None)
    parser.add_argument("--retry", default=None, metavar="CHECKPOINT_PATH")
    parser.add_argument("--retry-capped", default=None, metavar="CHECKPOINT_PATH")
    args = parser.parse_args()

    cfg = Config()
    if args.vllm_url:
        cfg.vllm_base_url = args.vllm_url
    if args.model:
        cfg.model_name = args.model
    if args.model_alias:
        cfg.model_alias = args.model_alias
    if args.project_root:
        cfg.project_root = Path(args.project_root)
    if args.output_dir:
        cfg.output_dir = args.output_dir
    if args.concurrency is not None:
        cfg.concurrency = args.concurrency
    if args.max_tokens is not None:
        cfg.max_tokens = args.max_tokens
    if args.temperature is not None:
        cfg.temperature = args.temperature
    if args.seed is not None:
        cfg.seed = args.seed
    if args.timeout is not None:
        cfg.request_timeout_sec = args.timeout
    if args.inter_delay is not None:
        cfg.inter_request_delay = args.inter_delay
    if args.early_stop is not None:
        cfg.early_stop_failures = args.early_stop
    if args.total_tables is not None:
        cfg.total_tables = args.total_tables
    if args.format_ratio is not None:
        cfg.format_ratio = args.format_ratio
    if args.context_window is not None:
        cfg.context_window = args.context_window
    if args.no_auto_detect_window:
        cfg.auto_detect_window = False
    if args.chunk_strategy is not None:
        cfg.chunk_strategy = args.chunk_strategy
    if args.header_zone_rows is not None:
        cfg.header_zone_rows = args.header_zone_rows
    if args.no_continuation:
        cfg.enable_continuation = False
    if args.max_continuation_rounds is not None:
        cfg.max_continuation_rounds = args.max_continuation_rounds
    if args.no_force_answer:
        cfg.force_answer_when_exhausted = False
    if args.no_disable_thinking:
        cfg.disable_thinking_supported = False
    if args.thinking_off_mode is not None:
        cfg.thinking_off_mode = args.thinking_off_mode
    if args.max_tokens_nonthinking is not None:
        cfg.max_tokens_nonthinking = args.max_tokens_nonthinking
    if args.no_tokenizer:
        cfg.use_tokenizer = False
    if args.cache_dir:
        cfg.cache_dir = args.cache_dir
    if args.extra_body:
        import json as _json
        cfg.extra_body = _json.loads(args.extra_body)
    if args.table_seed:
        cfg.table_seed_path = args.table_seed
    if args.log_level:
        cfg.log_level = args.log_level

    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, cfg.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()])

    collector = Collector(cfg)
    if args.retry:
        logging.info(f"=== RETRY MODE: {args.retry} ===")
        collector.run_retry(args.retry)
    elif args.retry_capped:
        logging.info(f"=== CAPPED RETRY MODE: {args.retry_capped} ===")
        collector.run_retry_capped(args.retry_capped)
    else:
        collector.run()


if __name__ == "__main__":
    main()
