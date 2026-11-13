// clang-format off
// %fsycl-host-only passes -fsycl-is-host straight to cc1 without -fsycl, so
// the driver does not forward -f[no-]sycl-rdc; pass the cc1 option directly.
// RUN: %clangxx %fsycl-host-only -Xclang -fno-gpu-rdc -E -dD %s -o %t.host
// RUN:                %clangxx -fsycl -fsycl-targets=spir64-unknown-unknown -fsycl-device-only -E -dD -fno-sycl-rdc %s -o %t.device.spirv
// RUN: %if cuda    %{ %clangxx -fsycl -fsycl-targets=nvptx64-nvidia-cuda    -fsycl-device-only -E -dD -fno-sycl-rdc %s -o %t.device.cuda %}
// RUN: %if hip     %{ %clangxx -fsycl -fsycl-targets=amdgcn-amd-amdhsa      -fsycl-device-only -E -dD -fno-sycl-rdc %s -o %t.device.hip  %}
//
// RUN: FileCheck --match-full-lines %s < %t.host --check-prefixes=HOST --implicit-check-not="#define SYCL_EXTERNAL"
// RUN:                FileCheck --match-full-lines %s < %t.device.spirv --check-prefixes=DEVICE-FULL-LINE --implicit-check-not="#define SYCL_EXTERNAL"
// RUN: %if cuda    %{ FileCheck --match-full-lines %s < %t.device.cuda  --check-prefixes=DEVICE-FULL-LINE --implicit-check-not="#define SYCL_EXTERNAL" %}
// RUN: %if hip     %{ FileCheck --match-full-lines %s < %t.device.hip   --check-prefixes=DEVICE-FULL-LINE --implicit-check-not="#define SYCL_EXTERNAL" %}
//
// Remove __DPCPP_SYCL_EXTERNAL to simplify regex for DEVICE prefix
// RUN:                sed 's|__DPCPP_SYCL_EXTERNAL||g' %t.device.spirv | FileCheck %s --check-prefixes=DEVICE
// RUN: %if cuda    %{ sed 's|__DPCPP_SYCL_EXTERNAL||g' %t.device.cuda  | FileCheck %s --check-prefixes=DEVICE %}
// RUN: %if hip     %{ sed 's|__DPCPP_SYCL_EXTERNAL||g' %t.device.hip   | FileCheck %s --check-prefixes=DEVICE %}
//
// With -fno-sycl-rdc, device code should not define or use SYCL_EXTERNAL
// DEVICE-FULL-LINE: #define __DPCPP_SYCL_EXTERNAL __attribute__((sycl_device))
// DEVICE-NOT:SYCL_EXTERNAL
//
// With -fno-sycl-rdc, host code should not define SYCL_EXTERNAL either
// HOST: #define __DPCPP_SYCL_EXTERNAL
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/bfloat16.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/oneapi/experimental/complex/complex.hpp>
