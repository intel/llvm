// This test checks for some basic Front End features for Native CPU:
// * Kernel name mangling
// * is-native-cpu module flag
// RUN: %clang_cc1 -fsycl-is-device -S -emit-llvm -internal-isystem %S/Inputs -fsycl-is-native-cpu -o %t.ll %s 
// RUN: FileCheck -input-file=%t.ll %s 

#include "sycl.hpp"
typedef __typeof__(sizeof(int)) size_t;
struct __nativecpu_state {
  alignas(16) size_t MGlobal_id[3];
};
#define __SYCL_HC_ATTRS                                                        \
  __attribute__((weak)) __attribute((alwaysinline))                            \
  [[intel::device_indirectly_callable]]

extern "C" __SYCL_HC_ATTRS __attribute((address_space(0))) size_t *
__dpcpp_nativecpu_global_id(__attribute((address_space(0)))
                            __nativecpu_state *s) {
  return &(s->MGlobal_id[0]);
}



using namespace sycl;
const size_t N = 10;

template <typename T>
class init_a;

template <typename T>
void test(queue myQueue) {
  {
    buffer<T, 1> a(range<1>{N});
    T test = 42;

    myQueue.submit([&](handler& cgh) {
      auto A = a.template get_access<access::mode::write>(cgh);
      cgh.parallel_for<init_a<T>>(range<1>{N}, [=](id<1> index) {
        A.use(index);
        A.use(test);
      });
    });
  }
}

void gen() {
  queue q;
  test<int>(q);
  test<float>(q);
}

// Check name mangling 
// CHECK-DAG: @_ZTS6init_aIiE_NativeCPUKernel_NativeCPUKernel({{.*}})
// CHECK-DAG: @_ZTS6init_aIfE_NativeCPUKernel_NativeCPUKernel({{.*}})

// Check Native CPU module flag
// CHECK-DAG: !{{[0-9]*}} = !{i32 1, !"is-native-cpu", i32 1}
