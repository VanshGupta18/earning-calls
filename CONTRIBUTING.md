# Contributing Guidelines

Welcome to the Multimodal Earnings Call Intelligence System. We follow a strict **Data Contract** approach to ensure parallel development.

## Project Structure
- `src/preprocessing`: Data ingestion and DB management (Devasya).
- `src/features`: Feature extraction pipelines (Vansh, Aadi).
- `src/modeling`: Training and benchmarking (Vansh).
- `src/evaluation`: Metrics and leakage control (Vansh).

## Development Workflow
1. **Branching**: Use feature branches (`feat/audio-extraction`).
2. **Data Contracts**: All new features must be saved as Parquet files in `data/processed/` and adhere to the agreed-upon schema in `implementation_plan.md`.
3. **Tests**: Add unit tests in `tests/` for any new extraction logic.
4. **Environment**: Ensure all dependencies are added to `requirements.txt`.

## Coding Standards
- Use type hints wherever possible.
- Use `polars` for data manipulation to ensure scalability.
- Document all extractors with docstrings explaining the features.
