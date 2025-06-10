// Make dump directory.
// RUN: rm -rf %t.spvdir && mkdir %t.spvdir

// RUN: %clangxx -fsycl -fsycl-dump-device-code=%t.spvdir %s

// Rename SPV file to explictly known filename.
// RUN: mv %t.spvdir/*.spv %t.spvdir/dump.spv

// Convert to LLVM IR.
// RUN: llvm-spirv -r --spirv-target-env=SPV-IR %t.spvdir/dump.spv
// RUN: llvm-dis %t.spvdir/dump.bc
// RUN: FileCheck --input-file=%t.spvdir/dump.ll %s

#include <cmath>
#include <sycl/detail/core.hpp>
#include <sycl/usm.hpp>

// CHECK: call spir_func float @_Z16__spirv_ocl_rintf(

using namespace sycl;

int main() {
  queue Q;

  float *Out = malloc_shared<float>(1, Q);
  Out[0] = 0.5f;

  try {
    Q.submit([&](handler &Cgh) {
      Cgh.parallel_for(nd_range<1>({1}, {1}),
                       [=](nd_item<1> Item) { *Out = std::rint(*Out); });
    });
  } catch (sycl::exception const &) {
  }

  free(Out, Q);
}
