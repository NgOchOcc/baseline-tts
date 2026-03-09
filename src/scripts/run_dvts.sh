BEAMSEARCH_DIR="scripts/Qwen2.5-7B/dvts"

echo "Starting all beamsearch experiments..."
echo "========================================"

# Run all .sh files in the beamsearch directory
for script in "$BEAMSEARCH_DIR"/run_*.sh; do
    if [ -f "$script" ]; then
        echo ""
        echo "Running: $(basename "$script")"
        echo "--------------------------------------"
        bash "$script"
    fi
    echo ""
done

echo "========================================"
echo "All beamsearch experiments completed!"
