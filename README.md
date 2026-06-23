# Table Header Detection — Experiment Pipeline

Модульный пайплайн для оценки распознавания заголовков таблиц LLM-моделями через
vLLM (OpenAI-совместимый API). Поддерживает датасеты RealHitBench
(`pubtables_complex_top500`, `maximum_viewpoint`, `table_normalization`),
форматы JSON/HTML и стратегии промптов zero/fewshot/reasoning.

## Структура

```
run.py                      запуск сбора данных (CLI)
analyze_results.py          кросс-модельный анализ (CI, парные сравнения)
requirements.txt
table_header_exp/
    config.py               единый источник конфигурации (Config, план эксперимента)
    datamodel.py            ApiResult (телеметрия запроса)
    parsing.py              разбор ответа модели и классификация ошибок API
    evaluation.py           чистые метрики (coord / type / soft-spanning / text)
    loading.py              загрузка таблиц и эталона
    prompts.py              сборка сообщений и бюджет max_tokens
    transport.py            BudgetController, async-вызовы, continuation
    chunking.py             header-aware чанкинг с абсолютными координатами
    persistence.py          инкрементальные чекпойнты, сводные метрики
    orchestrator.py         Collector — оркестрация прогона
```

## Установка

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## Запуск

Каталоги данных (`Get_500_Tables_from_PubTables/...`, `Convert_from_xlsx_to_Json/...`,
`Convert_from_json_to_html/...`) и `prompts/` ищутся относительно `--project-root`
(по умолчанию текущий каталог).

```bash
python run.py \
  --vllm-url http://127.0.0.1:8000/v1 \
  --model Qwen/Qwen3.5-9B --model-alias qwen35_9b \
  --project-root ~/okhotin/1/BetterExperimentAmpere \
  --output-dir results --concurrency 8 \
  --total-tables 1000 --format-ratio 50:50 \
  --temperature 0.0 --seed 42
```

Воспроизводимость между моделями — одинаковые таблицы:

```bash
python run.py ... --table-seed results/run_<...>/selected_tables.json
```

Дозапуск упавших / обрезанных:

```bash
python run.py ... --retry        results/run_<...>/checkpoints/checkpoint_latest.json
python run.py ... --retry-capped results/run_<...>/checkpoints/checkpoint_latest.json
```

Размер окна определяется автоматически из `/v1/models`; `--context-window` и
`--no-auto-detect-window` позволяют задать вручную.

## Анализ

```bash
python analyze_results.py results/run_A results/run_B --output-dir analysis --metric f1
```

`paired_model_comparison.csv` — парные дельты F1 между моделями на одинаковых
задачах с bootstrap-CI. `paired_format_comparison.csv` — сравнение JSON vs HTML
на пересечении успешно разобранных таблиц (без смещения из-за разной доступности
чанкинга); `comparison_by_model_format.csv` — сквозная картина с учётом отказов.
