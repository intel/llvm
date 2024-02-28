
// ONEAPI_DEVICE_SELECTOR, when called with an empty string, should be
// treated in the same manner when ONEAPI_DEVICE_SELECTOR is not set.

// RUN: %clangxx -fsycl -fsycl-targets=%{sycl_triple} %S/Inputs/trivial.cpp -o %t.out

// RUN: %{run-unfiltered-devices} %t.out > no_ods_output
// RUN: env ONEAPI_DEVICE_SELECTOR="" %{run-unfiltered-devices} %t.out > ods_empty_string_output
// RUN: diff no_ods_output ods_empty_string_output
