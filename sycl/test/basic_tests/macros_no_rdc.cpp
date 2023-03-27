// clang-format off
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple -fsycl-device-only -E -dD -fno-sycl-rdc %s -o %t.device
// RUN: %clangxx %fsycl-host-only -fno-sycl-rdc -E -dD %s -o %t.host
//
// RUN: FileCheck --match-full-lines %s < %t.device --check-prefixes=DEVICE-FULL-LINE --implicit-check-not="#define SYCL_EXTERNAL"
// RUN: FileCheck --match-full-lines %s < %t.host --check-prefixes=HOST
//
// Remove __DPCPP_SYCL_EXTERNAL to simplify regex for DEVICE prefix
// RUN: sed -i 's|__DPCPP_SYCL_EXTERNAL||g' %t.device
// RUN: FileCheck %s < %t.device --check-prefixes=DEVICE
//
// With -fno-sycl-rdc, device code should not define or use SYCL_EXTERNAL
// DEVICE-FULL-LINE: #define __DPCPP_SYCL_EXTERNAL __attribute__((sycl_device))
// DEVICE-NOT:SYCL_EXTERNAL
//
// With -fno-sycl-rdc, host code should have SYCL_EXTERNAL defined to empty
// HOST-DAG: #define SYCL_EXTERNAL
// HOST-DAG: #define __DPCPP_SYCL_EXTERNAL
#include <sycl/sycl.hpp>
#include "ext/oneapi/bfloat16.hpp"
#include "ext/intel/esimd.hpp"
#include "ext/oneapi/experimental/sycl_complex.hpp"
