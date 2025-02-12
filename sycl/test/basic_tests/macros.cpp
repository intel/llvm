// RUN: %clangxx %fsycl-host-only -dM -E %s -o %t.host
// RUN:                %clangxx -fsycl -fsycl-targets=spir64-unknown-unknown -fsycl-device-only -dM -E %s -o %t.device.spirv
// RUN: %if cuda    %{ %clangxx -fsycl -fsycl-targets=nvptx64-nvidia-cuda    -fsycl-device-only -dM -E %s -o %t.device.cuda %}
// RUN: %if hip     %{ %clangxx -fsycl -fsycl-targets=amdgcn-amd-amdhsa      -fsycl-device-only -dM -E %s -o %t.device.hip %}
//
// RUN: FileCheck %s < %t.host --check-prefixes=COMMON --implicit-check-not=__SPIRV
// RUN:                FileCheck %s < %t.device.spirv --check-prefixes=DEVICE,COMMON --implicit-check-not=__SPIRV
// RUN: %if cuda    %{ FileCheck %s < %t.device.cuda  --check-prefixes=DEVICE,COMMON --implicit-check-not=__SPIRV %}
// RUN: %if hip     %{ FileCheck %s < %t.device.hip   --check-prefixes=DEVICE,COMMON --implicit-check-not=__SPIRV %}
//
// FIXME: we should also check that we don't leak __SYCL* and SYCL* macro from
//        our header files.
//
// COMMON-DAG: #define SYCL_LANGUAGE_VERSION
// COMMON-DAG: #define __SYCL_COMPILER_VERSION
//
// DEVICE-DAG: #define SYCL_EXTERNAL
// DEVICE-DAG: #define __SYCL_DEVICE_ONLY__

#include <sycl/sycl.hpp>
