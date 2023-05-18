// REQUIRES: level_zero
// RUN: %clangxx -fsycl -fsycl-targets=%{sycl_triple} %S/Inputs/trivial.cpp -o %t.out

// Calling ONEAPI_DEVICE_SELECTOR with a backend and no device should result in
// an error.
// RUN: env ONEAPI_DEVICE_SELECTOR=level_zero %{run-unfiltered-devices} not %t.out
