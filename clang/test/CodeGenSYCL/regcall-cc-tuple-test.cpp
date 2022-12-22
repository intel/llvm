// RUN:  %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown -disable-llvm-passes -O0 -opaque-pointers -emit-llvm %s -o - | FileCheck %s
//
// Check that we use vector ABI in the arguments and return type for a __regcall esimd function with std::tuple
template <class T, int N> using raw_vector = T[N];

template <class T, int N>
struct simd {
  raw_vector<T, N> val;
};

using T1 = simd<double, 5>;
using T2 = simd<int, 20>;

namespace std {
template< class... Types >
class tuple {
  // Hardcode this as to not actually implement std::tuple
  T1 one;
  T2 two;
};
}
using MyTuple = std::tuple<T1, T2>;
// CHECK: define dso_local x86_regcallcc <30 x i32> @_Z16__regcall3__funcSt5tupleIJ4simdIdLi5EES0_IiLi20EEEE(<30 x i32> 
__attribute__((sycl_device))
__regcall MyTuple func(MyTuple a) __attribute__((sycl_explicit_simd)) {
   return a;
}

__attribute__((sycl_device)) int caller() {
  MyTuple val;
  func(val);
}
