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
    echo "### Cloning llvm-ci-perf-res ($SANITIZED_PERF_RES_GIT_REPO:$SANITIZED_PERF_RES_GIT_BRANCH) ###"
    mkdir -p "$(dirname "$SANITIZED_PERF_RES_PATH")"
    git clone -b $SANITIZED_PERF_RES_GIT_BRANCH https://github.com/$SANITIZED_PERF_RES_GIT_REPO $SANITIZED_PERF_RES_PATH
    [ "$?" -ne 0 ] && exit $? 
}

clone_compute_bench() {
    echo "### Cloning compute-benchmarks ($SANITIZED_COMPUTE_BENCH_GIT_REPO:$SANITIZED_COMPUTE_BENCH_GIT_BRANCH) ###"
    mkdir -p "$(dirname "$SANITIZED_COMPUTE_BENCH_PATH")"
    git clone -b $SANITIZED_COMPUTE_BENCH_GIT_BRANCH \
              --recurse-submodules https://github.com/$SANITIZED_COMPUTE_BENCH_GIT_REPO \
              $SANITIZED_COMPUTE_BENCH_PATH
    [ "$?" -ne 0 ] && exit $? 
}

build_compute_bench() {
    echo "### Building compute-benchmarks ($SANITIZED_COMPUTE_BENCH_GIT_REPO:$SANITIZED_COMPUTE_BENCH_GIT_BRANCH) ###"
    mkdir $SANITIZED_COMPUTE_BENCH_PATH/build && cd $SANITIZED_COMPUTE_BENCH_PATH/build &&
    # No reason to turn on ccache, if this docker image will be disassembled later on
    cmake .. -DBUILD_SYCL=ON -DBUILD_L0=OFF -DBUILD=OCL=OFF -DCCACHE_ALLOWED=FALSE
    # TODO enable mechanism for opting into L0 and OCL -- the concept is to
    # subtract OCL/L0 times from SYCL times in hopes of deriving SYCL runtime
    # overhead, but this is mostly an idea that needs to be mulled upon.

    if [ "$?" -eq 0 ]; then
        while IFS= read -r case; do
            # Skip lines starting with '#'
            [ "${case##\#*}" ] || continue

            if [ -n "$(printf "%s" "$case" | sed "s/[a-zA-Z_]*//g")" ]; then
                echo "Illegal characters in $TESTS_CONFIG."
                exit 1
            fi
            # TODO Sanitize this
            make "-j$SANITIZED_COMPUTE_BENCH_COMPILE_JOBS" "$case"
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
    [ ! -d "$SANITIZED_PERF_RES_PATH/$1" ] && return 1 # Directory doesn't exist
    file_count="$(find "$SANITIZED_PERF_RES_PATH/$1" -maxdepth 1 -type f | wc -l )"
    [ "$file_count" -lt "$SANITIZED_AVERAGE_MIN_THRESHOLD" ]
}

# Check for a regression via compare.py
#
# Usage: check_regression <relative path of output csv>
check_regression() {
    csv_relpath="$(dirname "$1")"
    csv_name="$(basename "$1")"
    if samples_under_threshold "$csv_relpath"; then
        echo "Not enough samples to construct a good average, performance\
 check skipped!"
        return 0 # Success status
    fi
    python "$DEVOPS_PATH/scripts/benchmarking/compare.py" \
        "$DEVOPS_PATH" "$csv_relpath" "$csv_name"
    return $?
}

# Move the results of our benchmark into the git repo, and save benchmark
# results to artifact archive
#
# Usage: cache <relative path of output csv>
cache() {
    mkdir -p "$(dirname "$SANITIZED_ARTIFACT_PASSING_CACHE/$1")" "$(dirname "$SANITIZED_PERF_RES_PATH/$1")"
    cp "$SANITIZED_ARTIFACT_OUTPUT_CACHE/$1" "$SANITIZED_ARTIFACT_PASSING_CACHE/$1"
    mv "$SANITIZED_ARTIFACT_OUTPUT_CACHE/$1" "$SANITIZED_PERF_RES_PATH/$1"
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
    mkdir -p "$SANITIZED_PERF_RES_PATH"
    
    echo "### Running and processing selected benchmarks ###"
    if [ -z "$TESTS_CONFIG" ]; then
        echo "Setting tests to run via cli is not currently supported."
        exit 1
    else
        rm "$SANITIZED_BENCHMARK_LOG_ERROR" "$SANITIZED_BENCHMARK_LOG_SLOW" 2> /dev/null
        mkdir -p "$(dirname "$SANITIZED_BENCHMARK_LOG_ERROR")" "$(dirname "$SANITIZED_BENCHMARK_LOG_SLOW")"
        # Loop through each line of enabled_tests.conf, but ignore lines in the
        # test config starting with #'s:
        grep "^[^#]" "$TESTS_CONFIG" | while read -r testcase; do
            if [ -n "$(printf "%s" "$testcase" | sed "s/[a-zA-Z_]*//g")" ]; then
                echo "Illegal characters in $TESTS_CONFIG."
                exit 1
            fi
            echo "# Running $testcase..."

            # The benchmark results git repo and this script's output both share
            # the following directory structure:
            #
            # /<device selector>/<runner>/<test name>
            #
            # Instead of specifying 2 paths with a slightly different root
            # folder name for every function we use, we can use a relative path
            # to represent the file in both folders.
            #
            # Figure out the relative path of our testcase result:
            test_dir_relpath="$DEVICE_SELECTOR_DIRNAME/$RUNNER/$testcase"
            output_csv_relpath="$test_dir_relpath/$testcase-$TIMESTAMP.csv"
			mkdir -p "$SANITIZED_ARTIFACT_OUTPUT_CACHE/$test_dir_relpath" # Ensure directory exists
            # TODO generate runner config txt if not exist

            output_csv="$SANITIZED_ARTIFACT_OUTPUT_CACHE/$output_csv_relpath"
            $SANITIZED_COMPUTE_BENCH_PATH/build/bin/$testcase --csv \
                --iterations="$SANITIZED_COMPUTE_BENCH_ITERATIONS" \
                    | tail +8 > "$output_csv"
                    # The tail +8 filters out header lines not in csv format

            exit_status="$?"
            if [ "$exit_status" -eq 0 ] && [ -s "$output_csv" ]; then 
                check_and_cache $output_csv_relpath
            else
                # TODO consider capturing stderr for logging
                echo "[ERROR] $testcase returned exit status $exit_status"
                echo "-- $testcase: error $exit_status" >> "$SANITIZED_BENCHMARK_LOG_ERROR"
            fi
        done
    fi
}

# Handle failures + produce a report on what failed
process_results() {
    fail=0
    if [ -s "$SANITIZED_BENCHMARK_LOG_SLOW" ]; then
        printf "\n### Tests performing over acceptable range of average: ###\n"
        cat "$SANITIZED_BENCHMARK_LOG_SLOW"
        echo ""
        fail=2
    fi
    if [ -s "$SANITIZED_BENCHMARK_LOG_ERROR" ]; then
        printf "\n### Tests that failed to run: ###\n"
        cat "$SANITIZED_BENCHMARK_LOG_ERROR"
        echo ""
        fail=1
    fi
    exit $fail
}

cleanup() {
    echo "### Cleaning up compute-benchmark builds from prior runs ###"
    rm -rf $SANITIZED_COMPUTE_BENCH_PATH
    rm -rf $SANITIZED_PERF_RES_PATH
    [ ! -z "$_exit_after_cleanup" ] && exit
}

load_configs() {
    # This script needs to know where the intel/llvm "/devops" directory is,
    # containing all the configuration files and the compare script.
    #
    # If this is not provided, this function tries to guess where the files
    # are based on how the script is called, and verifies that all necessary
    # configs and scripts are reachable. 

    # This benchmarking script is usually at:
    # 
    # /devops/scripts/benchmarking/benchmark.sh
    #
    # Derive /devops based on location of this script:
    [ -z "$DEVOPS_PATH" ] && DEVOPS_PATH="$(dirname "$0")/../.."

    TESTS_CONFIG="$(realpath $DEVOPS_PATH/benchmarking/enabled_tests.conf)"
    COMPARE_PATH="$(realpath $DEVOPS_PATH/scripts/benchmarking/compare.py)"
    LOAD_CONFIG_PY="$(realpath $DEVOPS_PATH/scripts/benchmarking/load_config.py)"

    for file in \
        "$TESTS_CONFIG" "$COMPARE_PATH" "$LOAD_CONFIG_PY"
    do
        if [ ! -f "$file" ]; then
            echo "Please provide path to /devops in DEVOPS_PATH."
            exit -1
        fi
    done

    $(python "$LOAD_CONFIG_PY" "$DEVOPS_PATH" config)
    $(python "$LOAD_CONFIG_PY" "$DEVOPS_PATH" constants)
}

#####

load_configs

COMPUTE_BENCH_COMPILE_FLAGS=""
CACHE_RESULTS="0"
TIMESTAMP="$(date +"$SANITIZED_TIMESTAMP_FORMAT")"

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
    echo "Please set CMPLR_ROOT first; it is needed by compute-benchmarks to build."
    exit 1
elif [ -z "$ONEAPI_DEVICE_SELECTOR" ]; then
    echo "Please set ONEAPI_DEVICE_SELECTOR first to specify which device to use."
    exit 1
elif [ -z "$RUNNER" ]; then
    echo "Please specify runner name using -n first; it is needed for storing/comparing benchmark results."
    exit 1
fi

# Make sure ONEAPI_DEVICE_SELECTOR doesn't try to enable multiple devices at the
# same time, or use specific device id's
_dev_sel_backend_re="$(echo "$SANITIZED_DEVICE_SELECTOR_ENABLED_BACKENDS" | sed 's/,/|/g')"
_dev_sel_device_re="$(echo "$SANITIZED_DEVICE_SELECTOR_ENABLED_DEVICES" | sed 's/,/|/g')"
_dev_sel_re="s/($_dev_sel_backend_re):($_dev_sel_device_re)//"
if [ -n "$(echo "$ONEAPI_DEVICE_SELECTOR" | sed -E "$_dev_sel_re")" ]; then
    echo "Unsupported ONEAPI_DEVICE_SELECTOR value: please ensure only one \
device is selected, and devices are not selected by indices."
    echo "Enabled backends: $SANITIZED_DEVICE_SELECTOR_ENABLED_BACKENDS"
    echo "Enabled device types: $SANITIZED_DEVICE_SELECTOR_ENABLED_DEVICES"
    exit 1
fi
# ONEAPI_DEVICE_SELECTOR values are not valid directory names in unix: this 
# value lets us use ONEAPI_DEVICE_SELECTOR as actual directory names 
DEVICE_SELECTOR_DIRNAME="$(echo "$ONEAPI_DEVICE_SELECTOR" | sed 's/:/-/')"

# Clean up and delete all cached files if specified:
[ ! -z "$_cleanup" ] && cleanup
# Clone and build only if they aren't already cached/deleted:
[ ! -d "$SANITIZED_PERF_RES_PATH"            ] && clone_perf_res
[ ! -d "$SANITIZED_COMPUTE_BENCH_PATH"       ] && clone_compute_bench
[ ! -d "$SANITIZED_COMPUTE_BENCH_PATH/build" ] && build_compute_bench
# Process benchmarks:
process_benchmarks
process_results
