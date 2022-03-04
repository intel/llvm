// REQUIRES: linux
// UNSUPPORTED: esimd_emulator_be
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: env SYCL_PI_TRACE=-1 %t.out 2>&1 | FileCheck %s

// Checks pi traces when libpi_esimd_emulator is not present
#include "library_loading_impl.h"
// CHECK: SYCL_PI_TRACE[-1]: dlopen(libpi_esimd_emulator.so) failed with <libpi_esimd_emulator.so: cannot open shared object file: No such file or directory>
