cargo clean
cargo build
cargo test
RUSTFLAGS="-C opt-level=3" maturin develop --release
python3 -m pytest tests/graph/test_community.py
python3 benchmark_community.py