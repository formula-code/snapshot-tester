CURR_DIR=$(pwd)
cd tests/test_repos/shapely_benchmarks/
rm -rf tests/test_repos/shapely_benchmarks/.snapshots
snapshot-tool capture .
snapshot-tool verify .
cd $CURR_DIR


CURR_DIR=$(pwd)
cd tests/test_repos/astropy_benchmarks/
rm -rf tests/test_repos/astropy_benchmarks/.snapshots
snapshot-tool capture .
snapshot-tool verify .
cd $CURR_DIR
