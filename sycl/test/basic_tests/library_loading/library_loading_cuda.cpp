// REQUIRES: linux
// UNSUPPORTED: cuda
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: env SYCL_PI_TRACE=-1 %t.out 2>&1 | FileCheck %s

// Checks pi traces when libpi_cuda is not present
#include "library_loading_impl.h"
// CHECK: SYCL_PI_TRACE[-1]: dlopen(libpi_cuda.so) failed with <libpi_cuda.so: cannot open shared object file: No such file or directory>
