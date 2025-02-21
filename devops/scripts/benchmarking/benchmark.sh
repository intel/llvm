#!/bin/sh

#
# benchmark.sh: Benchmark dpcpp using compute-benchmarks
#

usage () {
    >&2 echo "Usage: $0 <compute-benchmarks git repo> -t <runner type> [-B <compute-benchmarks build path>]
  -n  Github runner name -- Required
  -c  Clean up working directory
  -C  Clean up working directory and exit
  -s  Cache results

This script builds and runs benchmarks from compute-benchmarks."
    exit 1
}

# Ensures test cases read from enabled_tests.conf contains no malicious content
_validate_testname () {
    if [ -n "$(printf "%s" "$1" | sed "s/[a-zA-Z_]*//g")" ]; then
        echo "Illegal characters in $TEST_CONFIG. Permitted characters: a-zA-Z_"
        exit 1
    fi
}

clone_perf_res() {
    echo "### Cloning llvm-ci-perf-results ($SANITIZED_PERF_RES_GIT_REPO:$SANITIZED_PERF_RES_GIT_BRANCH) ###"
    git clone -b "$SANITIZED_PERF_RES_GIT_BRANCH" "https://github.com/$SANITIZED_PERF_RES_GIT_REPO" ./llvm-ci-perf-results
    [ "$?" -ne 0 ] && exit "$?"
}

clone_compute_bench() {
    echo "### Cloning compute-benchmarks ($SANITIZED_COMPUTE_BENCH_GIT_REPO:$SANITIZED_COMPUTE_BENCH_GIT_BRANCH) ###"
    git clone -b "$SANITIZED_COMPUTE_BENCH_GIT_BRANCH" \
              --recurse-submodules "https://github.com/$SANITIZED_COMPUTE_BENCH_GIT_REPO" \
              ./compute-benchmarks
    if [ ! -d "./compute-benchmarks" ]; then
        echo "Failed to clone compute-benchmarks."
        exit 1
    elif [ -n  "$SANITIZED_COMPUTE_BENCH_GIT_COMMIT" ]; then
        cd ./compute-benchmarks
        git checkout "$SANITIZED_COMPUTE_BENCH_GIT_COMMIT"
        if [ "$?" -ne 0 ]; then
            echo "Failed to get compute-benchmarks commit '$SANITIZED_COMPUTE_BENCH_GIT_COMMIT'."
            exit 1
        fi
        cd -
    fi
}

build_compute_bench() {
    echo "### Building compute-benchmarks ($SANITIZED_COMPUTE_BENCH_GIT_REPO:$SANITIZED_COMPUTE_BENCH_GIT_BRANCH) ###"
    mkdir ./compute-benchmarks/build && cd ./compute-benchmarks/build &&
    # No reason to turn on ccache, if this docker image will be disassembled later on
    cmake .. -DBUILD_SYCL=ON -DBUILD_L0=OFF -DBUILD=OCL=OFF -DCCACHE_ALLOWED=FALSE
    # TODO enable mechanism for opting into L0 and OCL -- the concept is to
    # subtract OCL/L0 times from SYCL times in hopes of deriving SYCL runtime
    # overhead, but this is mostly an idea that needs to be mulled upon.

    if [ "$?" -eq 0 ]; then
        while IFS= read -r case; do
            # Skip lines starting with '#'
            [ "${case##\#*}" ] || continue

            _validate_testname "$case"
            make "-j$SANITIZED_COMPUTE_BENCH_COMPILE_JOBS" "$case"
        done < "$TESTS_CONFIG"
    fi
    cd -
}

# Check if the number of samples for a given test case is less than a threshold
# set in benchmark-ci.conf
#
# Usage: <relative path of directory containing test case results>
samples_under_threshold () {
    # Directory doesn't exist, samples automatically under threshold
    [ ! -d "./llvm-ci-perf-results/$1" ] && return 0
    file_count="$(find "./llvm-ci-perf-results/$1" -maxdepth 1 -type f | wc -l )"
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
    mkdir -p "$(dirname ./artifact/passing_tests/$1)" "$(dirname ./artifact/failed_tests/$1)"
    cp "./artifact/failed_tests/$1" "./artifact/passing_tests/$1"
    mkdir -p "$(dirname ./llvm-ci-perf-results/$1)"
    mv "./artifact/failed_tests/$1" "./llvm-ci-perf-results/$1"
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
    echo "### Running and processing selected benchmarks ###"
    if [ -z "$TESTS_CONFIG" ]; then
        echo "Setting tests to run via cli is not currently supported."
        exit 1
    else
        rm ./artifact/benchmarks_errored.log ./artifact/benchmarks_failed.log 2> /dev/null
        mkdir -p ./artifact
        # Loop through each line of enabled_tests.conf, but ignore lines in the
        # test config starting with #'s:
        grep "^[^#]" "$TESTS_CONFIG" | while read -r testcase; do
            _validate_testname "$testcase"
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
			mkdir -p "./artifact/failed_tests/$test_dir_relpath" # Ensure directory exists

            # Tests are first placed in ./artifact/failed_tests, and are only
            # moved to passing_tests or the performance results repo if the
            # benchmark results are passing
            output_csv="./artifact/failed_tests/$output_csv_relpath"
            "./compute-benchmarks/build/bin/$testcase" --csv \
                --iterations="$SANITIZED_COMPUTE_BENCH_ITERATIONS" > "$output_csv"

            exit_status="$?"
            if [ "$exit_status" -eq 0 ] && [ -s "$output_csv" ]; then 
                # Filter out header lines not in csv format:
                tail +8 "$output_csv" > .tmp_res
                mv .tmp_res "$output_csv"
                check_and_cache $output_csv_relpath
            else
                echo "[ERROR] $testcase returned exit status $exit_status"
                echo "-- $testcase: error $exit_status" >> ./artifact/benchmarks_errored.log
            fi
        done
    fi
}

# Handle failures + produce a report on what failed
process_results() {
    fail=0
    if [ -s ./artifact/benchmarks_failed.log ]; then
        printf "\n### Tests performing over acceptable range of average: ###\n"
        cat ./artifact/benchmarks_failed.log
        echo ""
        fail=2
    fi
    if [ -s ./artifact/benchmarks_errored.log ]; then
        printf "\n### Tests that failed to run: ###\n"
        cat ./artifact/benchmarks_errored.log
        echo ""
        fail=1
    fi
    exit $fail
}

cleanup() {
    echo "### Cleaning up compute-benchmark builds from prior runs ###"
    rm -rf ./compute-benchmarks
    rm -rf ./llvm-ci-perf-results
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
    if [ -z "$(printf '%s' "$DEVOPS_PATH" | grep -oE '^[a-zA-Z0-9._\/-]+$')" ]; then
        echo "Bad DEVOPS_PATH, please specify DEVOPS_PATH variable."
        exit 1
    fi

    TESTS_CONFIG="$(realpath "$DEVOPS_PATH/benchmarking/enabled_tests.conf")"
    COMPARE_PATH="$(realpath "$DEVOPS_PATH/scripts/benchmarking/compare.py")"
    LOAD_CONFIG_PY="$(realpath "$DEVOPS_PATH/scripts/benchmarking/load_config.py")"

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
# Timestamp format is YYYYMMDD_HHMMSS
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

# CLI flags + overrides to configuration options:
while getopts "n:cCs" opt; do
    case "$opt" in
		n) 
        if [ -n "$(printf "%s" "$OPTARG" | sed "s/[a-zA-Z0-9_-]*//g")" ]; then
            echo "Illegal characters in runner name."
            exit 1
        fi
        RUNNER="$OPTARG"
        ;;
        # Cleanup status is saved in a var to ensure all arguments are processed before
        # performing cleanup
        c) _cleanup=1 ;;
        C) _cleanup=1 && _exit_after_cleanup=1 ;;
        s) CACHE_RESULTS=1;;
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
[ ! -d ./llvm-ci-perf-results     ] && clone_perf_res
[ ! -d ./compute-benchmarks       ] && clone_compute_bench
[ ! -d ./compute-benchmarks/build ] && build_compute_bench
# Process benchmarks:
process_benchmarks
process_results