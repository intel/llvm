// clang-format off
// RUN: %clangxx %fsycl-host-only -fno-sycl-rdc -E -dD %s -o %t.host
// RUN:                %clangxx -fsycl -fsycl-targets=spir64-unknown-unknown -fsycl-device-only -E -dD -fno-sycl-rdc %s -o %t.device.spirv
// RUN: %if cuda    %{ %clangxx -fsycl -fsycl-targets=nvptx64-nvidia-cuda    -fsycl-device-only -E -dD -fno-sycl-rdc %s -o %t.device.cuda %}
// RUN: %if hip_amd %{ %clangxx -fsycl -fsycl-targets=amdgcn-amd-amdhsa      -fsycl-device-only -E -dD -fno-sycl-rdc %s -o %t.device.hip  %}
//
// RUN: FileCheck --match-full-lines %s < %t.host --check-prefixes=HOST
// RUN:                FileCheck --match-full-lines %s < %t.device.spirv --check-prefixes=DEVICE-FULL-LINE --implicit-check-not="#define SYCL_EXTERNAL"
// RUN: %if cuda    %{ FileCheck --match-full-lines %s < %t.device.cuda  --check-prefixes=DEVICE-FULL-LINE --implicit-check-not="#define SYCL_EXTERNAL" %}
// RUN: %if hip_amd %{ FileCheck --match-full-lines %s < %t.device.hip   --check-prefixes=DEVICE-FULL-LINE --implicit-check-not="#define SYCL_EXTERNAL" %}
//
// Remove __DPCPP_SYCL_EXTERNAL to simplify regex for DEVICE prefix
// RUN:                sed 's|__DPCPP_SYCL_EXTERNAL||g' %t.device.spirv | FileCheck %s --check-prefixes=DEVICE
// RUN: %if cuda    %{ sed 's|__DPCPP_SYCL_EXTERNAL||g' %t.device.cuda  | FileCheck %s --check-prefixes=DEVICE %}
// RUN: %if hip_amd %{ sed 's|__DPCPP_SYCL_EXTERNAL||g' %t.device.hip   | FileCheck %s --check-prefixes=DEVICE %}
// RUN:
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
#include "ext/oneapi/experimental/complex/complex.hpp"
