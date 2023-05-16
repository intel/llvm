
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %S/Inputs/trivial.cpp -o %t.out
// RUN: env ONEAPI_DEVICE_SELECTOR="macaroni:*"" %t.out
// XFAIL: *

// Calling ONEAPI_DEVICE_SELECTOR with an illegal backend should result in an
// error.