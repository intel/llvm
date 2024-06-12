// REQUIRES: opencl, cpu, linux
// This test checks for correct behavior for shared library builds when new
// offload driver is enabled. Currently, new offload model supports only JIT.
// TODO: Expand the test once AOT support for new offload model is ready.
//
// RUN: %clangxx -fsycl -fsycl-targets=spir64 --offload-new-driver -DBUILD_LIB -fPIC -shared %s -o %T/lib%basename_t.so

// RUN: %clangxx -fsycl -fsycl-targets=spir64 --offload-new-driver -DFOO_FIRST -L%T %s -o %t.out -l%basename_t -Wl,-rpath=%T
// RUN: %{run} %t.out

// RUN: %clangxx -fsycl -fsycl-targets=spir64 --offload-new-driver -L%T %s -o %t.out -l%basename_t -Wl,-rpath=%T
// RUN: %{run} %t.out

#include "use_when_link.cpp"
