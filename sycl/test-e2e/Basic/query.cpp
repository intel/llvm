// RUN: %{build} -o %t.out
// RUN: %{run-unfiltered-devices} env ONEAPI_DEVICE_SELECTOR="*:gpu" %t.out

// REQUIRES: gpu-intel-dg1
#include "query.hpp"
