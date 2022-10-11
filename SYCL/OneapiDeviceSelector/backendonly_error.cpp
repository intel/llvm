// REQUIRES: level_zero
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %S/Inputs/trivial.cpp -o %t.out
// RUN: env ONEAPI_DEVICE_SELECTOR=level_zero %t.out
// XFAIL: *

// Calling ONEAPI_DEVICE_SELECTOR with a backend and no device should result in
// an error.