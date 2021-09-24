// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: env CreateMultipleSubDevices=2 EnableTimestampPacket=1 \
// RUN: NEOReadDebugKeys=1 SYCL_DEVICE_FILTER="gpu" %t.out

// UNSUPPORTED: gpu-intel-dg1,cuda,hip
#include "query.hpp"
