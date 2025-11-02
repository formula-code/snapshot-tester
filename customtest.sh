# CURR_DIR=$(pwd)
# cd tests/test_repos/shapely_benchmarks/
# rm -rf tests/test_repos/shapely_benchmarks/.snapshots
# snapshot-tool capture .
# snapshot-tool verify .
# cd $CURR_DIR

# harder tests.
CURR_DIR=$(pwd)
cd tests/test_repos/astropy_benchmarks/
rm -rf .snapshots
snapshot-tool list .
snapshot-tool capture . --filter ".*coordinates.*"
snapshot-tool verify . --filter ".*coordinates.*"
cd $CURR_DIR
