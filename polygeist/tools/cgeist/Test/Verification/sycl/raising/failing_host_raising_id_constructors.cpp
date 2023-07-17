// RUN: clang++ -O1 %s -S -emit-mlir -o - -fsycl -fsycl-raise-host -Xclang -opaque-pointers -w | FileCheck %s
// RUN: clang++ -O2 %s -S -emit-mlir -o - -fsycl -fsycl-raise-host -Xclang -opaque-pointers -w | FileCheck %s
// RUN: clang++ -O3 %s -S -emit-mlir -o - -fsycl -fsycl-raise-host -Xclang -opaque-pointers -w | FileCheck %s
// XFAIL: *

// COM: Copy and move constructors not being detected yet for Dimensions > 2.

// CHECK: sycl.host.constructor

#include <sycl/sycl.hpp>

template <typename... Args>
void keep(Args&&...);

template <int Dimensions>
void id(const sycl::id<Dimensions> &other) {
  sycl::id<Dimensions> id(other);
  keep(id);
}

template <int Dimensions>
void id(sycl::id<Dimensions> &&other) {
  sycl::id<Dimensions> id(std::move(other));
  keep(id);
}

template
void id<2>(const sycl::id<2> &);
template
void id<3>(const sycl::id<3> &);
template
void id<2>(sycl::id<2> &&);
template
void id<3>(sycl::id<3> &&);
