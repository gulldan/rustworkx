for so in rustworkx/rustworkx.cpython-*.so; do
    [ -e "$so" ] || continue
    mv -f "$so" "$so.stale"
done

cargo clean
cargo build
cargo test
RUSTFLAGS="-C opt-level=3" maturin develop --release

for so in rustworkx/rustworkx.cpython-*.so; do
    [ -e "$so" ] || continue
    mv -f "$so" "$so.stale"
done

uv run --group test python -m pytest tests/graph/test_community.py
uv run --group test python benchmark_community.py
