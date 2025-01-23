#!/bin/sh

#
# benchmark.sh: Benchmark dpcpp using compute-benchmarks
#

usage () {
    >&2 echo "Usage: $0 <compute-benchmarks git repo> -t <runner type> [-B <compute-benchmarks build path>]
  -n  Github runner name -- Required
  -B  Path to clone and build compute-benchmarks on
  -p  Path to compute-benchmarks (or directory to build compute-benchmarks in)
  -r  Github repo to use for compute-benchmarks origin, in format <org>/<name>
  -b  Git branch to use within compute-benchmarks
  -f  Compile flags passed into building compute-benchmarks
  -c  Clean up working directory
  -C  Clean up working directory and exit
  -s  Cache results

This script builds and runs benchmarks from compute-benchmarks."
    exit 1
}

clone_perf_res() {
    echo "### Cloning llvm-ci-perf-res ($PERF_RES_GIT_REPO:$PERF_RES_BRANCH) ###"
    mkdir -p "$(dirname $PERF_RES_PATH)"
    git clone -b $PERF_RES_BRANCH https://github.com/$PERF_RES_GIT_REPO $PERF_RES_PATH
    [ "$?" -ne 0 ] && exit $? 
}

clone_compute_bench() {
    echo "### Cloning compute-benchmarks ($COMPUTE_BENCH_GIT_REPO:$COMPUTE_BENCH_BRANCH) ###"
    mkdir -p "$(dirname $COMPUTE_BENCH_PATH)"
    git clone -b $COMPUTE_BENCH_BRANCH \
              --recurse-submodules https://github.com/$COMPUTE_BENCH_GIT_REPO \
              $COMPUTE_BENCH_PATH
    [ "$?" -ne 0 ] && exit $? 
}

build_compute_bench() {
    echo "### Building compute-benchmarks ($COMPUTE_BENCH_GIT_REPO:$COMPUTE_BENCH_BRANCH) ###"
    mkdir $COMPUTE_BENCH_PATH/build && cd $COMPUTE_BENCH_PATH/build &&
    # No reason to turn on ccache, if this docker image will be disassembled later on
    cmake .. -DBUILD_SYCL=ON -DBUILD_L0=OFF -DBUILD=OCL=OFF -DCCACHE_ALLOWED=FALSE # && cmake --build . $COMPUTE_BENCH_COMPILE_FLAGS
    # TODO enable mechanism for opting into L0 and OCL -- the concept is to
    # subtract OCL/L0 times from SYCL times in hopes of deriving SYCL runtime
    # overhead, but this is mostly an idea that needs to be mulled upon.

    if [ "$?" -eq 0 ]; then
        while IFS= read -r case; do
            # Skip lines starting with '#'
            [ "${case##\#*}" ] || continue
            make $COMPUTE_BENCH_COMPILE_FLAGS "$case"
        done < "$TESTS_CONFIG"
    fi
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

# Check if the number of samples for a given test case is less than a threshold
# set in benchmark-ci.conf
#
# Usage: <relative path of directory containing test case results>
samples_under_threshold () {
    [ ! -d "$PERF_RES_PATH/$1" ] && return 1 # Directory doesn't exist
    file_count="$(find "$PERF_RES_PATH/$1" -maxdepth 1 -type f | wc -l )"
    [ "$file_count" -lt "$AVERAGE_THRESHOLD" ]
}

# Check for a regression via compare.py
#
# Usage: check_regression <relative path of output csv>
check_regression() {
    csv_relpath="$(dirname $1)"
    csv_name="$(basename $1)"
    if samples_under_threshold "$csv_relpath"; then
        echo "Not enough samples to construct a good average, performance\
 check skipped!"
        return 0 # Success status
    fi
    BENCHMARKING_ROOT="$BENCHMARKING_ROOT" \
        python "$BENCHMARKING_ROOT/compare.py" "$csv_relpath" "$csv_name"
    return $?
}

# Move the results of our benchmark into the git repo
#
# Usage: cache <relative path of output csv>
cache() {
    mv "$OUTPUT_PATH/$1" "$PERF_RES_PATH/$1"
}

# Check for a regression + cache if no regression found
#
# Usage: check_and_cache <relative path of output csv>
check_and_cache() {
    echo "Checking $1..."
    if check_regression $1; then
        if [ "$CACHE_RESULTS" -eq "1" ]; then
            echo "Caching $1..."
            cache $1
        fi
    else
        [ "$CACHE_RESULTS" -eq "1" ] && echo "Regression found -- Not caching!"
    fi
}

# Run and process the results of each enabled benchmark in enabled_tests.conf
process_benchmarks() {
    mkdir -p "$PERF_RES_PATH"
    
    echo "### Running and processing selected benchmarks ###"
    if [ -z "$TESTS_CONFIG" ]; then
        echo "Setting tests to run via cli is not currently supported."
        exit 1
    else
        rm "$BENCHMARK_ERROR_LOG" "$BENCHMARK_SLOW_LOG" 2> /dev/null
        # Loop through each line of enabled_tests.conf, but ignore lines in the
        # test config starting with #'s:
        grep "^[^#]" "$TESTS_CONFIG" | while read -r testcase; do
            echo "# Running $testcase..."

            # The benchmark results git repo and this script's output both share
            # the following directory structure:
            #
            # /<device selector>/<runner>/<test name>
            #
            # Figure out the relative path of our testcase result in both
            # directories: 
            test_dir_relpath="$DEVICE_SELECTOR_DIRNAME/$RUNNER/$testcase"
			mkdir -p "$OUTPUT_PATH/$test_dir_relpath" # Ensure directory exists
            # TODO generate runner config txt if not exist
            output_csv_relpath="$test_dir_relpath/$testcase-$TIMESTAMP.csv"

            output_csv="$OUTPUT_PATH/$output_csv_relpath" # Real output path
            $COMPUTE_BENCH_PATH/build/bin/$testcase --csv \
                --iterations="$COMPUTE_BENCH_ITERATIONS" \
                    | tail +8 > "$output_csv"
                    # The tail +8 filters out header lines not in csv format

            exit_status="$?"
            if [ "$exit_status" -eq 0 ] && [ -s "$output_csv" ]; then 
                check_and_cache $output_csv_relpath
            else
                # TODO consider capturing stderr for logging
                echo "[ERROR] $testcase returned exit status $exit_status"
                echo "-- $testcase: error $exit_status" >> "$BENCHMARK_ERROR_LOG"
            fi
        done
    fi
}

# Handle failures + produce a report on what failed
process_results() {
    fail=0
    if [ -s "$BENCHMARK_SLOW_LOG" ]; then
        printf "\n### Tests performing over acceptable range of average: ###\n"
        cat "$BENCHMARK_SLOW_LOG"
        echo ""
        fail=2
    fi
    if [ -s "$BENCHMARK_ERROR_LOG" ]; then
        printf "\n### Tests that failed to run: ###\n"
        cat "$BENCHMARK_ERROR_LOG"
        echo ""
        fail=1
    fi
    exit $fail
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

    BENCHMARK_CI_CONFIG="$(realpath $BENCHMARKING_ROOT/benchmark-ci.conf)"
    TESTS_CONFIG="$(realpath $BENCHMARKING_ROOT/enabled_tests.conf)"
    COMPARE_PATH="$(realpath $BENCHMARKING_ROOT/compare.py)"

    for file in "$BENCHMARK_CI_CONFIG" "$TESTS_CONFIG" "$COMPARE_PATH"; do
        if [ ! -f "$file" ]; then
            echo "$(basename $file) not found, please provide path to BENCHMARKING_ROOT."
            exit -1
        fi
    done

    . $BENCHMARKING_ROOT/utils.sh
    load_all_configs "$BENCHMARK_CI_CONFIG"
}

#####

COMPUTE_BENCH_COMPILE_FLAGS=""
CACHE_RESULTS="0"
TIMESTAMP="$(date +"$TIMESTAMP_FORMAT")"

load_configs

# CLI flags + overrides to configuration options:
while getopts "p:b:r:f:n:cCs" opt; do
    case $opt in
        p) COMPUTE_BENCH_PATH=$OPTARG ;;
        r) COMPUTE_BENCH_GIT_REPO=$OPTARG ;;
        b) COMPUTE_BENCH_BRANCH=$OPTARG ;;
        f) COMPUTE_BENCH_COMPILE_FLAGS=$OPTARG ;;
		n) RUNNER=$OPTARG ;;
        # Cleanup status is saved in a var to ensure all arguments are processed before
        # performing cleanup
        c) _cleanup=1 ;;
        C) _cleanup=1 && _exit_after_cleanup=1 ;;
        s) CACHE_RESULTS="1";;
        \?) usage ;;
    esac
done

# Check all necessary variables exist:
if [ -z "$CMPLR_ROOT" ]; then
    echo "Please set \$CMPLR_ROOT first; it is needed by compute-benchmarks to build."
    exit 1
elif [ -z "$ONEAPI_DEVICE_SELECTOR" ]; then
    echo "Please set \$ONEAPI_DEVICE_SELECTOR first to specify which device to use."
    exit 1
elif [ -z "$RUNNER" ]; then
    echo "Please specify runner name using -n first; it is needed for storing/comparing benchmark results."
    exit 1
fi

# Make sure ONEAPI_DEVICE_SELECTOR doesn't try to enable multiple devices at the
# same time, or use specific device id's
_dev_sel_backend_re="$(sed 's/,/|/g' <<< "$DEVICE_SELECTOR_ENABLED_BACKENDS")"
_dev_sel_device_re="$(sed 's/,/|/g' <<< "$DEVICE_SELECTOR_ENABLED_DEVICES")"
_dev_sel_re="s/($_dev_sel_backend_re):($_dev_sel_device_re)//"
if [ -n "$(sed -E "$_dev_sel_re" <<< "$ONEAPI_DEVICE_SELECTOR" )" ]; then
    echo "Unsupported \$ONEAPI_DEVICE_SELECTOR value: please ensure only one \
device is selected, and devices are not selected by indices."
    echo "Enabled backends: $DEVICE_SELECTOR_ENABLED_BACKENDS"
    echo "Enabled device types: $DEVICE_SELECTOR_ENABLED_DEVICES"
    exit 1
fi
# ONEAPI_DEVICE_SELECTOR values are not valid directory names in unix: this 
# value lets us use ONEAPI_DEVICE_SELECTOR as actual directory names 
DEVICE_SELECTOR_DIRNAME="$(sed 's/:/-/' <<< "$ONEAPI_DEVICE_SELECTOR")"

# Clean up and delete all cached files if specified:
[ ! -z "$_cleanup" ] && cleanup
# Clone and build only if they aren't already cached/deleted:
[ ! -d "$PERF_RES_PATH"            ] && clone_perf_res
[ ! -d "$COMPUTE_BENCH_PATH"       ] && clone_compute_bench
[ ! -d "$COMPUTE_BENCH_PATH/build" ] && build_compute_bench
# Process benchmarks:
process_benchmarks
process_results
