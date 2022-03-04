// REQUIRES: linux
// UNSUPPORTED: hip_be
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: env SYCL_PI_TRACE=-1 %t.out 2>&1 | FileCheck %s

// Checks pi traces when libpi_hip is not present
#include "library_loading_impl.h"
// CHECK: SYCL_PI_TRACE[-1]: dlopen(libpi_hip.so) failed with <libpi_hip.so: cannot open shared object file: No such file or directory>
