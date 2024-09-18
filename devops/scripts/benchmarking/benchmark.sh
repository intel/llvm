#!/bin/sh

#
# benchmark.sh: Benchmark dpcpp using compute-benchmarks
#

usage () {
    >&2 echo "Usage: $0 <compute-benchmarks git repo> [-B <compute-benchmarks build path>]
  -B  Path to clone and build compute-benchmarks on
  -p  Path to compute-benchmarks (or directory to build compute-benchmarks in)
  -r  Git repo url to use for compute-benchmarks origin
  -b  Git branch to use within compute-benchmarks
  -f  Compile flags passed into building compute-benchmarks
  -c  _cleanup=1 ;;
  -C  _cleanup=1 && _exit_after_cleanup=1 ;;

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
    cmake .. -DBUILD_SYCL=ON  -DCCACHE_ALLOWED=FALSE && cmake --build . $COMPUTE_BENCH_COMPILE_FLAGS
    # No reason to turn on ccache, if this docker image will be disassembled later on
    #compute_bench_build_stat=$?
    cd -
    #[ "$compute_bench_build_stat" -ne 0 ] && exit $compute_bench_build_stat 
}

# print_bench_res() {
#     # Usage: print_bench_res <benchmark output .csv file> <benchmark status code> <summary file>
#     if [ ! -s $1 ]; then
#         printf "NO OUTPUT! (Status $2)\n" | tee -a $3
#         return  # Do not proceed if file is empty
#     fi
#     
#     get_csv_col_index $1 run-time-mean
#     tmp_run_time_mean_i=$tmp_csv_col_i
#     get_csv_col_index $1 run-time-median
#     tmp_run_time_median_i=$tmp_csv_col_i
#     get_csv_col_index $1 run-time-throughput
#     tmp_run_time_throughput_i=$tmp_csv_col_i
# 
#     # `sycl-bench` output seems to like inserting the header multiple times.
#     # Here we cache the header to make sure it prints only once:
#     tmp_header_title="$(cat $1 | head -n 1 | sed 's/^\# Benchmark name/benchmark/')"
#     tmp_result="$(cat $1 | grep '^[^\#]')"
# 
#     printf "%s\n%s" "$tmp_header_title" "$tmp_result"                  \
#         | awk -F',' -v me="$tmp_run_time_mean_i"                       \
#                     -v md="$tmp_run_time_median_i"                     \
#                     -v th="$tmp_run_time_throughput_i"                 \
#             '{printf "%-57s %-13s %-15s %-20s\n", $1, $me, $md, $th }' \
#         | tee -a $3   # Print to summary file
# }

###
STATUS_SUCCESS=0
STATUS_ERROR=1
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
}

cache() {
    mv "$2" "$PERF_RES_PATH/$1/"
}

# Check for a regression, and cache if no regression found
check_and_cache() {
    echo "Checking $testcase..."
    if check_regression $1 $2; then
        if [ "$CACHE_RESULTS" -eq "1" ]; then
            echo "Caching $testcase..."
            cache $1 $2
        fi
    else
        if [ "$CACHE_RESULTS" -eq "1" ]; then
            echo "Not caching!"
        fi
    fi
}

process_benchmarks() {
    mkdir -p "$PERF_RES_PATH"
    
    echo "### Running and processing selected benchmarks ###"
    if [ -z "$TESTS_CONFIG" ]; then
        echo "Setting tests to run via cli is not currently supported."
        exit $STATUS_ERROR
    else
        rm "$BENCHMARK_ERROR_LOG" "$BENCHMARK_SLOW_LOG" 2> /dev/null
        # Ignore lines in the test config starting with #'s
        grep "^[^#]" "$TESTS_CONFIG" | while read -r testcase; do
            echo "# Running $testcase..."
            test_csv_output="$OUTPUT_PATH/$testcase-$TIMESTAMP.csv"
            $COMPUTE_BENCH_PATH/build/bin/$testcase --csv | tail +8 > "$test_csv_output"
            # The tail +8 filters out initial debug prints not in csv format
            if [ "$?" -eq 0 ] && [ -s "$test_csv_output" ]; then 
                check_and_cache $testcase $test_csv_output
            else
                # TODO consider capturing error for logging
                echo "ERROR @ $test_case"
                echo "-- $testcase: error $?" >> "$BENCHMARK_ERROR_LOG"
            fi
        done
    fi
}

process_results() {
    if [ -s "$BENCHMARK_SLOW_LOG" ]; then
        printf "\n### Tests performing over acceptable range of average: ###\n"
        cat "$BENCHMARK_SLOW_LOG"
        echo ""
    fi
    if [ -s "$BENCHMARK_ERROR_LOG" ]; then
        printf "\n### Tests that failed to run: ###\n"
        cat "$BENCHMARK_ERROR_LOG"
        echo ""
    fi
    [ ! -s "$BENCHMARKING_SLOW_LOG" ] && [ ! -s "$BENCHMARK_ERROR_LOG" ]
}

cleanup() {
    echo "### Cleaning up compute-benchmark builds from prior runs ###"
    rm -rf $COMPUTE_BENCH_PATH
    rm -rf $PERF_RES_PATH
    [ ! -z "$_exit_after_cleanup" ] && exit
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

COMPUTE_BENCH_COMPILE_FLAGS=""
CACHE_RESULTS="0"
TIMESTAMP="$(date +"$TIMESTAMP_FORMAT")"

# CLI overrides to configuration options
while getopts "p:b:r:f:cCs" opt; do
    case $opt in
        p) COMPUTE_BENCH_PATH=$OPTARG ;;
        r) COMPUTE_BENCH_GIT_REPO=$OPTARG ;;
        b) COMPUTE_BENCH_BRANCH=$OPTARG ;;
        f) COMPUTE_BENCH_COMPILE_FLAGS=$OPTARG ;;
        # Cleanup status is saved in a var to ensure all arguments are processed before
        # performing cleanup
        c) _cleanup=1 ;;
        C) _cleanup=1 && _exit_after_cleanup=1 ;;
        s) CACHE_RESULTS="1";;
        \?) usage ;;
    esac
done

if [ -z "$CMPLR_ROOT" ]; then
    echo "Please set \$CMPLR_ROOT first; it is needed by compute-benchmarks to build."
    exit 1
fi
[ ! -z "$_cleanup" ] && cleanup

[ ! -d "$PERF_RES_PATH"            ] && clone_perf_res
[ ! -d "$COMPUTE_BENCH_PATH"       ] && clone_compute_bench
[ ! -d "$COMPUTE_BENCH_PATH/build" ] && build_compute_bench
process_benchmarks
process_results