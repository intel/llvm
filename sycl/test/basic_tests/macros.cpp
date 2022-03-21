// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple -fsycl-device-only -dM -E %s -o %t.device
// RUN: %clangxx %fsycl-host-only -dM -E %s -o %t.host
//
// RUN: FileCheck %s < %t.device --check-prefixes=DEVICE,COMMON \
// RUN:     --implicit-check-not=__SPIRV
// RUN: FileCheck %s < %t.host --check-prefixes=COMMON \
// RUN:      --implicit-check-not=__SPIRV
//
// COMMON-DAG: #define SYCL_LANGUAGE_VERSION
//
// DEVICE-DAG: #define SYCL_EXTERNAL
// DEVICE-DAG: #define __SYCL_DEVICE_ONLY__

#include <CL/sycl.hpp>
