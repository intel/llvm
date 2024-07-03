#!/bin/sh

# sycl-bench.sh: Benchmark dpcpp using sycl-bench

usage () {
    >&2 echo "Usage: $0 <sycl-bench git repo> [-B <sycl-bench build path>]
  -B  Path to clone and build sycl-bench on

This script builds and runs benchmarks from sycl-bench."
    exit 1
}

clone() {
    mkdir -p $SYCL_BENCH_PATH
    git clone $SYCL_BENCH_GIT_REPO $SYCL_BENCH_PATH || return $?
}

build() {
    cd $SYCL_BENCH_PATH
    cmake -DSYCL_IMPL=dpcpp -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_RUNTIME_OUTPUT_DIRECTORY=./bin -S . -B ./build &&
    cmake --build ./build || return $?
    cd -
}

get_csv_col_index() {
    # Determine the index of a column in a CSV given its title
    # Usage: get_csv_col_index <benchmark output .csv file> <column name>
    tmp_csv_col_i="$(cat "$1" | head -n 1 | grep -o "^.*$2," | grep -o ',' | wc -l)"
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

# run sycl bench step
run() {
    TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
    mkdir "$SYCL_BENCH_PATH/build/bench-$TIMESTAMP/"
    tmp_summary_file="$SYCL_BENCH_PATH/build/bench-$TIMESTAMP/summary.txt" 

    for file in $SYCL_BENCH_PATH/build/bin/*; do
        # TODO -size should not be always 256, caution
        tmp_bench_output="$SYCL_BENCH_PATH/build/bench-$TIMESTAMP/$(basename $file).csv"
        tmp_bench_log="$SYCL_BENCH_PATH/build/bench-$TIMESTAMP/$(basename $file).log"

        tmp_err="0"
        printf "\n### Results for $(basename $file) ###\n" | tee -a $tmp_summary_file
        # The pipe here suppresses errors in a way that doesn't stop github actions:
        $file --output=$tmp_bench_output --no-verification --size=256 2> "$tmp_bench_log" || tmp_err=$?
        print_bench_res $tmp_bench_output $tmp_err $tmp_summary_file
        # Remove log if nothing logged
        [ ! -s "$tmp_bench_log" ] && rm "$tmp_bench_log" || cat "$tmp_bench_log" | tee -a $tmp_summary_file
    done

    # Export timestamp for later use
    [ -f "$GITHUB_OUTPUT" ] && echo TIMESTAMP=$TIMESTAMP >> $GITHUB_OUTPUT
}

compress() {
    tar -I gzip -cf "$SYCL_BENCH_PATH/build/bench-$TIMESTAMP.tar.gz" -C "$SYCL_BENCH_PATH/build/bench-$TIMESTAMP" .
    if [ -f "$SYCL_BENCH_PATH/build/bench-$TIMESTAMP.tar.gz" ] && [ -f "$GITHUB_OUTPUT" ]; then
        echo BENCHMARK_RESULTS="$SYCL_BENCH_PATH/build/bench-$TIMESTAMP.tar.gz" >> $GITHUB_OUTPUT
    fi
}

cleanup() {
    rm -r $SYCL_BENCH_PATH
}


[ "$#" -lt "1" ] && usage

SYCL_BENCH_GIT_REPO="$1"; shift
SYCL_BENCH_PATH="./sycl-bench"
while getopts "B:" opt; do
    case $opt in
        B)  SYCL_BENCH_PATH=$OPTARG ;;
        \?) usage ;;
    esac
done

clone && build && run && compress
