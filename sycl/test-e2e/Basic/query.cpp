// RUN: %{build} -o %t.out
// RUN: %{run-unfiltered-devices} env ONEAPI_DEVICE_SELECTOR="*:gpu" %t.out

// REQUIRES: arch-intel_gpu_dg1
#include "query.hpp"
