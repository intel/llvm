// RUN: clang++ -O1 %s -S -emit-mlir -o - -fsycl -fsycl-raise-host -Xclang -opaque-pointers -w | FileCheck %s
// RUN: clang++ -O2 %s -S -emit-mlir -o - -fsycl -fsycl-raise-host -Xclang -opaque-pointers -w | FileCheck %s
// RUN: clang++ -O3 %s -S -emit-mlir -o - -fsycl -fsycl-raise-host -Xclang -opaque-pointers -w | FileCheck %s
// XFAIL: *

// COM: We fail to detect 2-D nd_range constructor as the nd_range allocation is GEPed using an `i8` element type.

// CHECK: sycl.host.constructor({{.*}}) {type = !sycl_nd_range_2_}

#include <sycl/sycl.hpp>

template <typename... Args>
void keep(Args&&...);

template <int Dimensions>
void nd_range(sycl::range<Dimensions> globalSize,
              sycl::range<Dimensions> localSize) {
  sycl::nd_range<Dimensions> ndr(globalSize, localSize);
  keep(ndr);
}

template <int Dimensions>
void nd_range_offset(sycl::range<Dimensions> globalSize,
                     sycl::range<Dimensions> localSize,
                     sycl::id<Dimensions> offset) {
  sycl::nd_range<Dimensions> ndr(globalSize, localSize, offset);
  keep(ndr);
}

template void nd_range(sycl::range<2>, sycl::range<2>);
template void nd_range_offset(sycl::range<2>, sycl::range<2>, sycl::id<2>);
