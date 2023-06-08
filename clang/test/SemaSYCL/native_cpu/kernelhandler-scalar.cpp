// RUN: %clangxx -fsycl-device-only  -fsycl-targets=native_cpu -Xclang -fsycl-int-header=%t.h -S -o %t.ll %s 
// RUN: FileCheck -input-file=%t.h.hc %s 
// Compiling generated main integration header to check correctness, -fsycl option used to find required includes
// RUN: %clangxx -fsycl -D __SYCL_NATIVE_CPU__ -c -x c++ %t.h
#include <CL/sycl.hpp>

#include <cstdlib>
#include <iostream>

using namespace cl::sycl;

const size_t N = 10;

template <typename T>
class init_a;

template <typename T>
bool test(queue myQueue) {
  {
    buffer<T, 1> a(range<1>{N});
    T test = rand() % 10;

    myQueue.submit([&](handler& cgh) {
      auto A = a.template get_access<access::mode::write>(cgh);
      cgh.parallel_for<init_a<T>>(range<1>{N}, [=](id<1> index) {
        A[index] = test;
      });
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
  if(!(res1 && res2 && res3 && res4)) {
    return 1;
  }
  return 0;
}



// CHECK:extern "C" void _Z6init_aIiEsubhandler(const sycl::detail::NativeCPUArgDesc *MArgs, __nativecpu_state *state);
// CaHECK:extern "C" void _Z6init_aIjEsubhandler(const sycl::detail::NativeCPUArgDesc *MArgs, __nativecpu_state *state);
// CaHECK:extern "C" void _Z6init_aIfEsubhandler(const sycl::detail::NativeCPUArgDesc *MArgs, __nativecpu_state *state);
// CaHECK:extern "C" void _Z6init_aIdEsubhandler(const sycl::detail::NativeCPUArgDesc *MArgs, __nativecpu_state *state);


