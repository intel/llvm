// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: env SYCL_DEVICE_FILTER="gpu" %t.out

// REQUIRES: gpu-intel-dg1
#include "query.hpp"
