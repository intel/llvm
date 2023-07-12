// RUN: %clangxx -fsycl-device-only  -fsycl-targets=native_cpu -Xclang -fsycl-int-header=%t.h -S -o %t.ll %s
// RUN: FileCheck -input-file=%t.h.hc %s
// RUN: FileCheck -input-file=%t.ll %s --check-prefix=CHECK-LL
// Compiling generated main integration header to check correctness, -fsycl
// option used to find required includes
// RUN: %clangxx -fsycl -D __SYCL_NATIVE_CPU__ -c -x c++ %t.h
#include <CL/sycl.hpp>

#include <iostream>

using namespace cl::sycl;

const size_t N = 10;

template <typename T> class init_a;

template <typename T> bool test(queue myQueue) {
  {
    buffer<T, 1> a(range<1>{N});
    T test = 42;

    myQueue.submit([&](handler &cgh) {
      auto A = a.template get_access<access::mode::write>(cgh);
      cgh.parallel_for<init_a<T>>(range<1>{N},
                                  [=](id<1> index) { A[index] = test; });
    });

    auto A = a.template get_access<access::mode::read>();
    std::cout << "Result:" << std::endl;
    for (size_t i = 0; i < N; i++) {
      if (A[i] != test) {
        std::cout << "ERROR\n";
        return false;
      }
    }
  }

  std::cout << "Good computation!" << std::endl;
  return true;
}

int main() {
  queue q;
  int res1 = test<int>(q);
  int res2 = test<unsigned>(q);
  int res3 = test<float>(q);
  int res4 = test<double>(q);
  if (!(res1 && res2 && res3 && res4)) {
    return 1;
  }
  return 0;
}

//CHECK: extern "C" void _ZTS6init_aIiE_NativeCPUKernel(void *, void *, int, __nativecpu_state *);
//CHECK: inline static void _ZTS6init_aIiE_NativeCPUKernelsubhandler(const sycl::detail::NativeCPUArgDesc *MArgs, __nativecpu_state *state) {
//CHECK-NEXT:   void* arg0 = MArgs[0].getPtr();
//CHECK-NEXT:   void* arg3 = MArgs[3].getPtr();
//CHECK-NEXT:   int arg4 = *(int*)MArgs[4].getPtr();
//CHECK-NEXT:   _ZTS6init_aIiE_NativeCPUKernel(arg0, arg3, arg4, state);
//CHECK-NEXT: };

//CHECK: extern "C" void _ZTS6init_aIjE_NativeCPUKernel(void *, void *, unsigned int, __nativecpu_state *);
//CHECK: inline static void _ZTS6init_aIjE_NativeCPUKernelsubhandler(const sycl::detail::NativeCPUArgDesc *MArgs, __nativecpu_state *state) {
//CHECK-NEXT:   void* arg0 = MArgs[0].getPtr();
//CHECK-NEXT:   void* arg3 = MArgs[3].getPtr();
//CHECK-NEXT:   unsigned int arg4 = *(unsigned int*)MArgs[4].getPtr();
//CHECK-NEXT:   _ZTS6init_aIjE_NativeCPUKernel(arg0, arg3, arg4, state);
//CHECK-NEXT: };

//CHECK: extern "C" void _ZTS6init_aIfE_NativeCPUKernel(void *, void *, float, __nativecpu_state *);
//CHECK: inline static void _ZTS6init_aIfE_NativeCPUKernelsubhandler(const sycl::detail::NativeCPUArgDesc *MArgs, __nativecpu_state *state) {
//CHECK-NEXT:   void* arg0 = MArgs[0].getPtr();
//CHECK-NEXT:   void* arg3 = MArgs[3].getPtr();
//CHECK-NEXT:   float arg4 = *(float*)MArgs[4].getPtr();
//CHECK-NEXT:   _ZTS6init_aIfE_NativeCPUKernel(arg0, arg3, arg4, state);
//CHECK-NEXT: };

//CHECK: extern "C" void _ZTS6init_aIdE_NativeCPUKernel(void *, void *, double, __nativecpu_state *);
//CHECK: inline static void _ZTS6init_aIdE_NativeCPUKernelsubhandler(const sycl::detail::NativeCPUArgDesc *MArgs, __nativecpu_state *state) {
//CHECK-NEXT:   void* arg0 = MArgs[0].getPtr();
//CHECK-NEXT:   void* arg3 = MArgs[3].getPtr();
//CHECK-NEXT:   double arg4 = *(double*)MArgs[4].getPtr();
//CHECK-NEXT:   _ZTS6init_aIdE_NativeCPUKernel(arg0, arg3, arg4, state);
//CHECK-NEXT: };

// CHECK-LL-DAG: @_ZTS6init_aIiE_NativeCPUKernel(ptr {{.*}}%0, ptr {{.*}}%1, i32 {{.*}}%2, ptr {{.*}}%3){{.*}}!kernel_arg_type ![[TYPE1:[0-9]*]]
// CHECK-LL-DAG: @_ZTS6init_aIjE_NativeCPUKernel(ptr {{.*}}%0, ptr {{.*}}%1, i32 {{.*}}%2, ptr {{.*}}%3){{.*}}!kernel_arg_type ![[TYPE2:[0-9]*]]
// CHECK-LL-DAG: @_ZTS6init_aIfE_NativeCPUKernel(ptr {{.*}}%0, ptr {{.*}}%1, float {{.*}}%2, ptr {{.*}}%3){{.*}}!kernel_arg_type ![[TYPE3:[0-9]*]]
// CHECK-LL-DAG: @_ZTS6init_aIdE_NativeCPUKernel(ptr {{.*}}%0, ptr {{.*}}%1, double {{.*}}%2, ptr {{.*}}%3){{.*}}!kernel_arg_type ![[TYPE4:[0-9]*]]
// CHECK-LL-DAG: ![[TYPE1]] = !{!"int*", !"sycl::range<>", !"sycl::range<>", !"sycl::id<1>", !"int"}
// CHECK-LL-DAG: ![[TYPE2]] = !{!"uint*", !"sycl::range<>", !"sycl::range<>", !"sycl::id<1>", !"unsigned int"}
// CHECK-LL-DAG: ![[TYPE3]] = !{!"float*", !"sycl::range<>", !"sycl::range<>", !"sycl::id<1>", !"float"}
// CHECK-LL-DAG: ![[TYPE4]] = !{!"double*", !"sycl::range<>", !"sycl::range<>", !"sycl::id<1>", !"double"}
