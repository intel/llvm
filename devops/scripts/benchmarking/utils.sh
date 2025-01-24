#!/bin/sh

#
# utils.sh: Utilities for benchmarking scripts
#

_sanitize_configs() {
    # Remove all characters other than characters specified in sed expression:
    echo "$1" | sed 's/[^a-zA-Z0-9_.,:/%-]//g'
}

# Sanitize + load all known configuration options
# Usage: load_all_configs <config file>
load_all_configs() {
    # Strict loading of configuration options by name:
    while IFS='=' read -r key value; do
        sanitized_value=$(_sanitize_configs "$value")
        case "$key" in
            'PERF_RES_GIT_REPO')
                export PERF_RES_GIT_REPO="$sanitized_value" ;;
            'PERF_RES_BRANCH')
                export PERF_RES_BRANCH="$sanitized_value" ;;
            'PERF_RES_PATH')
                export PERF_RES_PATH="$sanitized_value" ;;
            'COMPUTE_BENCH_GIT_REPO')
                export COMPUTE_BENCH_GIT_REPO="$sanitized_value" ;;
            'COMPUTE_BENCH_BRANCH')
                export COMPUTE_BENCH_BRANCH="$sanitized_value" ;;
            'COMPUTE_BENCH_PATH')
                export COMPUTE_BENCH_PATH="$sanitized_value" ;;
            'COMPUTE_BENCH_COMPILE_FLAGS')
                export COMPUTE_BENCH_COMPILE_FLAGS="$sanitized_value" ;;
            'COMPUTE_BENCH_ITERATIONS')
                export COMPUTE_BENCH_ITERATIONS="$sanitized_value" ;;
            'OUTPUT_CACHE')
                export OUTPUT_CACHE="$sanitized_value" ;;
            'PASSING_CACHE')
                export PASSING_CACHE="$sanitized_value" ;;
            'METRICS_VARIANCE')
                export METRICS_VARIANCE="$sanitized_value" ;;
            'METRICS_RECORDED')
                export METRICS_RECORDED="$sanitized_value" ;;
            'AVERAGE_THRESHOLD')
                export AVERAGE_THRESHOLD="$sanitized_value" ;;
            'AVERAGE_CUTOFF_RANGE')
                export AVERAGE_CUTOFF_RANGE="$sanitized_value" ;;
            'TIMESTAMP_FORMAT')
                export TIMESTAMP_FORMAT="$sanitized_value" ;;
            'BENCHMARK_SLOW_LOG')
                export BENCHMARK_SLOW_LOG="$sanitized_value" ;;
            'BENCHMARK_ERROR_LOG')
                export BENCHMARK_ERROR_LOG="$sanitized_value" ;;
            'RUNNER_TYPES')
                export RUNNER_TYPES="$sanitized_value" ;;
            'DEVICE_SELECTOR_ENABLED_BACKENDS')
                export DEVICE_SELECTOR_ENABLED_BACKENDS="$sanitized_value" ;;
            'DEVICE_SELECTOR_ENABLED_DEVICES')
                export DEVICE_SELECTOR_ENABLED_DEVICES="$sanitized_value" ;;
            'ARTIFACT_PATH')
                export ARTIFACT_PATH="$sanitized_value" ;;
            #*) echo "Unknown key: $sanitized_key" ;;
        esac
    done < "$1"
}

# Sanitize + load a single configuration value
# Usage: load_single_config <config file> <config name>
load_single_config() {
    _val="$(_sanitize_configs "$(grep "^$2=" "$1" | sed "s/^$2=//")")"
    export "$2=$_val"
}

# TODO: Do I want this?
# # Print a single configuration value
# # Usage: print_single_config <config file> <config name>
# print_single_config() {
#     _sanitize_configs "$(grep "^$2=" "$1" | sed "s/^$2=//")"
# }