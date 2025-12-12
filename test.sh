cargo clean
cargo build
cargo test
RUSTFLAGS="-C opt-level=3" maturin develop --release
uv run python -m pytest tests/graph/test_community.py
uv run python benchmark_community.py