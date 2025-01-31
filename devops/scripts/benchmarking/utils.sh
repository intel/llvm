#!/bin/sh

#
# utils.sh: Utilities for benchmarking scripts
#

# Usage: _sanitize_configs <field value>
_sanitize_configs() {
    # Trim quotes if any
    trim_quotes="$(printf "%s" "$2" | tr -d "\n" | sed 's/^"//; s/"$//')"
    check_illegal_chars="$(printf "%s" "$trim_quotes" | sed 's/[a-zA-Z0-9_.,:/%-]//g')"

    if [ -n "$check_illegal_chars" ]; then
        # Throw if unallowed characters are spotted
        printf ""
    else
        # Return the trimmed string
        printf "%s" "$trim_quotes"
    fi
}

_preprocess_config() {
    # Remove comments
    _tmp1="$(mktemp)"
    grep '^[^#]' "$1" > "$_tmp1" 
    # Skip values intended for python
    _tmp2="$(mktemp)"
    grep -E -v '^METRICS_(VARIANCE|RECORDED)' "$_tmp1" > "$_tmp2"
    rm "$_tmp1"
    # Return
    echo "$_tmp2"
}

# Sanitize + load all known configuration options
# Usage: load_config_options <config file>
load_config_options() {
    processed_config="$(_preprocess_config $1)"
    # Strict loading of configuration options by name:
    while IFS='=' read -r key value; do
        sanitized_value=$(_sanitize_configs "$key" "$value")
        if [ -z "$sanitized_value" ]; then
            echo "Bad configuration value for $key: $value"
            echo "Ensure $value is within character range [a-zA-Z0-9_.,:/%-]."
            exit 1
        fi

        case "$key" in
            'COMPUTE_BENCH_COMPILE_FLAGS')
                export COMPUTE_BENCH_COMPILE_FLAGS="$sanitized_value" ;;
            'COMPUTE_BENCH_ITERATIONS')
                export COMPUTE_BENCH_ITERATIONS="$sanitized_value" ;;
            'AVERAGE_THRESHOLD')
                export AVERAGE_THRESHOLD="$sanitized_value" ;;
            'AVERAGE_CUTOFF_RANGE')
                export AVERAGE_CUTOFF_RANGE="$sanitized_value" ;;
            'DEVICE_SELECTOR_ENABLED_BACKENDS')
                export DEVICE_SELECTOR_ENABLED_BACKENDS="$sanitized_value" ;;
            'DEVICE_SELECTOR_ENABLED_DEVICES')
                export DEVICE_SELECTOR_ENABLED_DEVICES="$sanitized_value" ;;
        esac
    done < "$processed_config"
}

# Sanitize + load all (known) constants from the configuration file
# Usage: load_config_constants <config file>
load_config_constants() {
    processed_config="$(_preprocess_config $1)"
    # Strict loading of configuration options by name:
    while IFS='=' read -r key value; do
        sanitized_value=$(_sanitize_configs "$key" "$value")
        if [ -z "$sanitized_value" ]; then
            echo "Bad configuration value for $key: $value"
            echo "Ensure $value is within character range [a-zA-Z0-9_.,:/%-]."
            exit 1
        fi

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
            'OUTPUT_CACHE')
                export OUTPUT_CACHE="$sanitized_value" ;;
            'ARTIFACT_PATH')
                export ARTIFACT_PATH="$sanitized_value" ;;
            'PASSING_CACHE')
                export PASSING_CACHE="$sanitized_value" ;;
            'TIMESTAMP_FORMAT')
                export TIMESTAMP_FORMAT="$sanitized_value" ;;
            'BENCHMARK_SLOW_LOG')
                export BENCHMARK_SLOW_LOG="$sanitized_value" ;;
            'BENCHMARK_ERROR_LOG')
                export BENCHMARK_ERROR_LOG="$sanitized_value" ;;
        esac
    done < "$processed_config"
}

# # Sanitize + load a single configuration value
# # Usage: load_single_config <config file> <config name>
# load_single_config() {
#     _val="$(_sanitize_configs "$(grep "^$2=" "$1" | sed "s/^$2=//")")"
#     export "$2=$_val"
# }
