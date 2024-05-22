// RUN: %clangxx -fsycl-device-only -O2 -g -fexceptions  -fsycl-targets=native_cpu -Xclang -sycl-std=2020 -mllvm -sycl-opt -S -emit-llvm  -o %t_temp.ll %s
// RUN: %clangxx -mllvm -sycl-native-cpu-backend -S -emit-llvm -o - %t_temp.ll | FileCheck %s

// Checks that the subhandler is correctly emitted in the module
#include <sycl/sycl.hpp>

#include <cstdlib>
#include <iostream>
const size_t N = 10;

template <typename T> class init_a;
class Test1;
using namespace sycl;

template <typename T> void gen_test(queue myQueue) {
  buffer<float, 1> a(range<1>{N});
  const T test = rand() % 10;

  myQueue
      .submit([&](handler &cgh) {
        auto A = a.get_access<access::mode::write>(cgh);
        cgh.parallel_for<init_a<T>>(range<1>{N},
                                    [=](id<1> index) { A[index] = test; });
      })
      .wait();
}

template <typename name, typename Func>
__attribute__((sycl_kernel)) void launch(const Func &kernelFunc) {
  kernelFunc();
}

void test() {
  queue q;
  gen_test<int>(q);
  //CHECK:  define void @_ZTS6init_aIiE.SYCLNCPU(ptr %{{.*}}, ptr addrspace(1) {{.*}}) #{{.*}} {
  //CHECK:  %{{.*}} = getelementptr %{{.*}}, ptr %{{.*}}, i64 {{.*}}
  //CHECK:  %{{.*}} = load ptr addrspace(1), ptr %{{.*}}
  //CHECK:  %{{.*}} = getelementptr %{{.*}}, ptr %{{.*}}, i64 {{.*}}
  //CHECK:  %{{.*}} = load ptr, ptr %{{.*}}
  //CHECK:  %{{.*}} = getelementptr %{{.*}}, ptr %{{.*}}, i64 {{.*}}
  //CHECK:  %{{.*}} = load ptr, ptr %{{.*}}
  //CHECK:  %{{.*}} = load i32, ptr %{{.*}}
  //CHECK:  call void @_ZTS6init_aIiE.NativeCPUKernel(ptr {{.*}}, ptr {{.*}}, i32 {{.*}}, ptr {{.*}})
  //CHECK:  ret void
  //CHECK:}
  gen_test<float>(q);
  //CHECK:  define void @_ZTS6init_aIfE.SYCLNCPU(ptr %{{.*}}, ptr addrspace(1) {{.*}}) #{{.*}} {
  //CHECK:  %{{.*}} = getelementptr %{{.*}}, ptr %{{.*}}, i64 {{.*}}
  //CHECK:  %{{.*}} = load ptr addrspace(1), ptr %{{.*}}
  //CHECK:  %{{.*}} = getelementptr %{{.*}}, ptr %{{.*}}, i64 {{.*}}
  //CHECK:  %{{.*}} = load ptr, ptr %{{.*}}, align 8
  //CHECK:  %{{.*}} = getelementptr %{{.*}}, ptr %{{.*}}, i64 {{.*}}
  //CHECK:  %{{.*}} = load ptr, ptr %{{.*}}
  //CHECK:  %{{.*}} = load float, ptr %{{.*}}
  //CHECK:  call void @_ZTS6init_aIfE.NativeCPUKernel(ptr {{.*}}, ptr {{.*}}, float {{.*}}, ptr {{.*}})
  //CHECK:  ret void
  //CHECK:}

  // Check that subhandler is emitted correctly for kernels with no
  // args:deviceQueue.submit([&](sycl::handler &h) {
  sycl::accessor<int, 1, sycl::access::mode::write> acc;
  q.submit([&](sycl::handler &h) {
    h.parallel_for<Test1>(range<1>(1), [=](sycl::id<1> id) {
      acc[id[0]]; // all kernel arguments are removed
    });
  });
  //CHECK:define void @_ZTS5Test1.SYCLNCPU(ptr %{{.*}}, ptr addrspace(1) %[[STATE2:.*]]) #{{.*}} {
  //CHECK:       call void @_ZTS5Test1.NativeCPUKernel(ptr addrspace(1) %[[STATE2]])
  //CHECK-NEXT:  ret void
  //CHECK-NEXT:}

  launch<class TestKernel>([]() {});
  //CHECK:define void @_ZTSZ4testvE10TestKernel.SYCLNCPU(ptr %{{.*}}, ptr addrspace(1) %[[STATE3:.*]]) #{{.*}} {
  //CHECK:       call void @_ZTSZ4testvE10TestKernel.NativeCPUKernel(ptr addrspace(1) %[[STATE3]])
  //CHECK-NEXT:  ret void
  //CHECK-NEXT:}
}
