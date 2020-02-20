// RUN: %clang_cc1 -fcxx-exceptions -triple spir64 -fsycl-is-device -Wno-return-type -verify -fsyntax-only -std=c++17 %s

/*
    ensuring that the SYCL diagnostics that are typically deferred, correctly emit 
*/

struct S {
  virtual void foo() {}
};

int calledFromKernel(int a){
  // expected-error@+1 {{zero-length arrays are not permitted in C++}}
  int MalArray[0];  

  // expected-error@+1 {{__float128 is not supported on this target}}
  __float128 malFloat = 40;  

  S mal;
  // expected-error@+1 {{SYCL kernel cannot call a virtual function}}
  mal.foo();  

  return a + 20;
}

inline namespace cl {
namespace sycl {
class queue {
public:
  template <typename T> void submit(T CGF) {}
};

template <int I> class id {};

template <int I> class range {};

class handler {
public:
  template <typename KernelName, typename KernelType, int Dims>
  __attribute__((sycl_kernel)) void kernel_parallel_for(KernelType kernelFunc) {
    kernelFunc(id<1>{});
  }
  template <typename KernelName, typename KernelType, int Dims>
  void parallel_for(range<Dims> NWI, KernelType kernelFunc) {
    kernel_parallel_for<KernelName, KernelType, Dims>(kernelFunc);
  }
};
}
}

int main(int argc, char **argv) {

  cl::sycl::range<1> numOfItems;
  cl::sycl::queue deviceQueue;

  deviceQueue.submit([&](cl::sycl::handler &cgh) {
    cgh.parallel_for<class AName>(numOfItems, [=](cl::sycl::id<1> wiID) {
      // expected-error@+1 {{zero-length arrays are not permitted in C++}}
      int BadArray[0]; 

      // expected-error@+1 {{__float128 is not supported on this target}}
      __float128 badFloat = 40; // this SHOULD  trigger a diagnostic

      S s;
      // expected-error@+1 {{zero-length arrays are not permitted in C++}}
      s.foo();   

      calledFromKernel(10);
    });
  });

  return 0;
}