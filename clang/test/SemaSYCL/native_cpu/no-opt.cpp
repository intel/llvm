// The sycl-native-cpu helper header is always named <sycl-int-header>.hc
// RUN: %clangxx -fsycl -fsycl-native-cpu -O0 -o %t.bc %s 
// This test currently fails because the PrepareSYCLNativeCPU pass doesn't support non optimized code (in particular non-inlined code)
// XFAIL: *

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


