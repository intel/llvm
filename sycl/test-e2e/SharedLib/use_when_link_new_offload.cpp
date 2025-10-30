// REQUIRES: opencl, cpu, linux
// This test checks for correct behavior for shared library builds when new
// offload driver is enabled. Currently, new offload model supports only JIT.
// TODO: Expand the test once AOT support for new offload model is ready.
//
// RUN: rm -rf %t.dir; mkdir -p %t.dir
// RUN: %clangxx -fsycl -fsycl-targets=spir64 --offload-new-driver -DBUILD_LIB -fPIC -shared %s -o %t.dir/lib%basename_t.so

// RUN: %clangxx -fsycl -fsycl-targets=spir64 --offload-new-driver -DFOO_FIRST -L%t.dir %s -o %t.out -l%basename_t -Wl,-rpath=%t.dir
// RUN: %{run} %t.out

// RUN: %clangxx -fsycl -fsycl-targets=spir64 --offload-new-driver -L%t.dir %s -o %t.out -l%basename_t -Wl,-rpath=%t.dir
// RUN: %{run} %t.out

#include "use_when_link.cpp"
