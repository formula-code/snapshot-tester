# Run from repository root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Do this for all repos
for repo in tests/test_repos/*_benchmarks/; do
    echo "Processing repository: $repo"
    cd "$repo" || { echo "Failed to change directory to $repo"; exit 1; }
    echo "Removing existing .snapshots directory..."
    rm -rf .snapshots
    echo "Listing benchmarks..."
    snapshot-tool list . 2>&1 | head -20
    echo "Capturing snapshots..."
    snapshot-tool capture .
    echo "Verifying snapshots..."
    snapshot-tool verify .
    echo "Completed processing for $repo"
    cd "$SCRIPT_DIR" || { echo "Failed to change directory to $SCRIPT_DIR"; exit 1; }
done
