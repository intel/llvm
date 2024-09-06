#!/bin/sh

#
# benchmark.sh: Benchmark dpcpp using compute-benchmarks
#

# TODO fix
usage () {
    >&2 echo "Usage: $0 <compute-benchmarks git repo> [-B <compute-benchmarks build path>]
  -B  Path to clone and build compute-benchmarks on

This script builds and runs benchmarks from compute-benchmarks."
    exit 1
}

clone_perf_res() {
    echo "### Cloning llvm-ci-perf-res ($PERF_RES_GIT_REPO:$PERF_RES_BRANCH) ###"
    mkdir -p "$(dirname $PERF_RES_PATH)"
    git clone -b $PERF_RES_BRANCH $PERF_RES_GIT_REPO $PERF_RES_PATH
    [ "$?" -ne 0 ] && exit $? 
}

clone_compute_bench() {
    echo "### Cloning compute-benchmarks ($COMPUTE_BENCH_GIT_REPO:$COMPUTE_BENCH_BRANCH) ###"
    mkdir -p "$(dirname $COMPUTE_BENCH_PATH)"
    git clone -b $COMPUTE_BENCH_BRANCH \
              --recurse-submodules $COMPUTE_BENCH_GIT_REPO \
              $COMPUTE_BENCH_PATH
    [ "$?" -ne 0 ] && exit $? 
}

build_compute_bench() {
    echo "### Building compute-benchmarks ($COMPUTE_BENCH_GIT_REPO:$COMPUTE_BENCH_BRANCH) ###"
    mkdir $COMPUTE_BENCH_PATH/build && cd $COMPUTE_BENCH_PATH/build &&
    cmake .. -DBUILD_SYCL=ON && cmake --build .
    compute_bench_build_stat=$?
    cd -
    [ "$compute_bench_build_stat" -ne 0 ] && exit $compute_bench_build_stat 
}

print_bench_res() {
    # Usage: print_bench_res <benchmark output .csv file> <benchmark status code> <summary file>
    if [ ! -s $1 ]; then
        printf "NO OUTPUT! (Status $2)\n" | tee -a $3
        return  # Do not proceed if file is empty
    fi
    
    get_csv_col_index $1 run-time-mean
    tmp_run_time_mean_i=$tmp_csv_col_i
    get_csv_col_index $1 run-time-median
    tmp_run_time_median_i=$tmp_csv_col_i
    get_csv_col_index $1 run-time-throughput
    tmp_run_time_throughput_i=$tmp_csv_col_i

    # `sycl-bench` output seems to like inserting the header multiple times.
    # Here we cache the header to make sure it prints only once:
    tmp_header_title="$(cat $1 | head -n 1 | sed 's/^\# Benchmark name/benchmark/')"
    tmp_result="$(cat $1 | grep '^[^\#]')"

    printf "%s\n%s" "$tmp_header_title" "$tmp_result"                  \
        | awk -F',' -v me="$tmp_run_time_mean_i"                       \
                    -v md="$tmp_run_time_median_i"                     \
                    -v th="$tmp_run_time_throughput_i"                 \
            '{printf "%-57s %-13s %-15s %-20s\n", $1, $me, $md, $th }' \
        | tee -a $3   # Print to summary file
}

###
STATUS_SUCCESS=0
STATUS_FAILED=1
###

samples_under_threshold () {
    mkdir -p $1
    file_count="$(find $1 -maxdepth 1 -type f | wc -l )"
    [ "$file_count" -lt "$AVERAGE_THRESHOLD" ]
}

check_regression() {
    if samples_under_threshold "$PERF_RES_PATH/$1"; then
        echo "Not enough samples to construct an average, performance check skipped!"
        return $STATUS_SUCCESS
    fi
    BENCHMARKING_ROOT="$BENCHMARKING_ROOT" python "$BENCHMARKING_ROOT/compare.py" "$1" "$2"
    return $?
    # return $STATUS_FAILED
}

cache() {
    mv "$2" "$PERF_RES_PATH/$1/"
}

# Check for a regression, and cache if no regression found
check_and_cache() {
    echo "Checking $testcase..."
    if check_regression $1 $2; then
        echo "Caching $testcase..."
        cache $1 $2
    else
        echo "Not caching!"
    fi
}

process_benchmarks() {
    TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
    mkdir -p "$PERF_RES_PATH"
    
    echo "### Running and processing selected benchmarks ###"
    if [ -z "$TESTS_CONFIG" ]; then
        echo "Setting tests to run via cli is not currently supported."
        exit $STATUS_FAILED
    else
        # Ignore lines in the test config starting with #'s
        grep "^[^#]" "$TESTS_CONFIG" | while read -r testcase; do
            echo "# Running $testcase..."
            test_csv_output="$OUTPUT_PATH/$testcase-$TIMESTAMP.csv"
            $COMPUTE_BENCH_PATH/build/bin/$testcase --csv | tail +8 > "$test_csv_output"
            # The tail +8 filters out initial debug prints not in csv format
            if [ "$?" -eq 0 ] && [ -s "$test_csv_output" ]; then 
                check_and_cache $testcase $test_csv_output
            else
                echo "ERROR @ $test_case"
            fi
        done
    fi
}

cleanup() {
    rm -r $COMPUTE_BENCH_PATH
}

load_configs() {
    # This script needs to know where the "BENCHMARKING_ROOT" directory is,
    # containing all the configuration files and the compare script.
    #
    # If this is not provided, this function tries to guess where the files
    # are based on how the script is called, and verifies that all necessary
    # configs and scripts are reachable. 
    [ -z "$BENCHMARKING_ROOT" ] && BENCHMARKING_ROOT="$(dirname $0)"

    BENCHMARK_CI_CONFIG="$BENCHMARKING_ROOT/benchmark-ci.conf"
    TESTS_CONFIG="$BENCHMARKING_ROOT/enabled_tests.conf"
    COMPARE_PATH="$BENCHMARKING_ROOT/compare.py"

    for file in "$BENCHMARK_CI_CONFIG" "$TESTS_CONFIG" "$COMPARE_PATH"; do
        if [ ! -f "$file" ]; then
            echo "$(basename $file) not found, please provide path to BENCHMARKING_ROOT."
            exit -1
        fi
    done

    . $BENCHMARK_CI_CONFIG
}

load_configs

# CLI overrides to configuration options
while getopts "p:b:r:" opt; do
    case $opt in
        p) COMPUTE_BENCH_PATH=$OPTARG ;;
        r) COMPUTE_BENCH_GIT_REPO=$OPTARG ;;
        b) COMPUTE_BENCH_BRANCH=$OPTARG ;;
        \?) usage ;;
    esac
done

[ ! -d "$PERF_RES_PATH"            ] && clone_perf_res
[ ! -d "$COMPUTE_BENCH_PATH"       ] && clone_compute_bench
[ ! -d "$COMPUTE_BENCH_PATH/build" ] && build_compute_bench
process_benchmarks