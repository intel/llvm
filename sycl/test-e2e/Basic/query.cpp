// RUN: %{build} -o %t.out
// RUN: env ONEAPI_DEVICE_SELECTOR="*:gpu" %t.out

// REQUIRES: gpu-intel-dg1
#include "query.hpp"
