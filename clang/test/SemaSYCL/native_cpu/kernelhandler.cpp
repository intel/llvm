// Checks that kernelhandler and sunhandler are emitted in the integration headers.
// The sycl-native-cpu helper header is always named <sycl-int-header>.hc
// RUN: %clangxx -fsycl-device-only -fsycl-native-cpu -Xclang -fsycl-int-header=%t.h -o %t.bc %s 
// RUN: FileCheck -input-file=%t.h %s --check-prefix=CHECK-H
// RUN: FileCheck -input-file=%t.h.hc %s --check-prefix=CHECK-HC
// Compiling generated main integration header to check correctness, -fsycl option used to find required includes 
// RUN: %clangxx -fsycl -D __SYCL_NATIVE_CPU__ -c -x c++ %t.h

#include "sycl.hpp"
class Test1;
int main() {
  sycl::queue deviceQueue;
  sycl::accessor<int, 1, sycl::access::mode::write> acc;
  sycl::range<1> r(1);
  deviceQueue.submit([&](sycl::handler& h){
        
        h.parallel_for<Test1>(r, [=](sycl::id<1> id){
             acc[id[0]] = 42;
            });
      });
}


// CHECK-H: template <> struct KernelInfo<::Test1> {
// CHECK-H-NEXT:   static constexpr bool is_native_cpu = 1;
// CHECK-H-NEXT:   static inline void NCPUKernelHandler(const std::vector<sycl::detail::NativeCPUArgDesc>& MArgs, nativecpu_state* s) {
// CHECK-H-NEXT:     _Z5Test1subhandler(MArgs, s);
// CHECK-H-NEXT:   }



//CHECK-HC: #pragma once
//CHECK-HC-NEXT: #include <sycl/detail/native_cpu.hpp>
//CHECK-HC:extern "C" void _Z5Test1(void *, void *, nativecpu_state *);
//CHECK-HC:inline static void _Z5Test1subhandler(const std::vector<sycl::detail::NativeCPUArgDesc>& MArgs, nativecpu_state *state) {
//CHECK-HC-NEXT:  void* arg0 = MArgs[0].getPtr();
//CHECK-HC-NEXT:  void* arg3 = MArgs[3].getPtr();
//CHECK-HC-NEXT:  _Z5Test1(arg0, arg3, state);
//CHECK-HC-NEXT:};
