// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: env CreateMultipleSubDevices=2 EnableTimestampPacket=1 \
// RUN: NEOReadDebugKeys=1 ONEAPI_DEVICE_SELECTOR="*:gpu" %t.out

// UNSUPPORTED: gpu-intel-dg1,cuda,hip
// Temporarily disable on L0 due to fails in CI
// UNSUPPORTED: level_zero
#include "query.hpp"
