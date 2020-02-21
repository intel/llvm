// RUN: %clang_cc1 -fcxx-exceptions -triple spir64 -fsycl-is-device -Wno-return-type -verify -fsyntax-only %s

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

  // not sure if 'no virtual function' is a _deferred_ diagnostic, testing anyway 
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

/*
  template used to specialize a function that contains a lambda that should
  result in a deferred diagnostic being emitted.
  HOWEVER, this is not working presently.  
  TODO: re-test after new deferred diagnostic system is merged. 
        restore the "FIX!!" tests below
*/
template <typename T>
void setup_sycl_operation(const T VA[]) {
    
  cl::sycl::range<1> numOfItems;
  cl::sycl::queue deviceQueue;

  deviceQueue.submit([&](cl::sycl::handler &cgh) {
    cgh.parallel_for<class AName>(numOfItems, [=](cl::sycl::id<1> wiID) {
      // FIX!!  expected-error@+1 {{zero-length arrays are not permitted in C++}}
      int OverlookedBadArray[0]; 
                       
      // FIX!!   expected-error@+1 {{__float128 is not supported on this target}}
      __float128 overlookedBadFloat = 40; 
 
    });
  });
}

int main(int argc, char **argv) {

  // --- direct lambda testing ---
  cl::sycl::range<1> numOfItems;
  cl::sycl::queue deviceQueue;

  deviceQueue.submit([&](cl::sycl::handler &cgh) {
    cgh.parallel_for<class AName>(numOfItems, [=](cl::sycl::id<1> wiID) {
      // expected-error@+1 {{zero-length arrays are not permitted in C++}}
      int BadArray[0]; 

      // expected-error@+1 {{__float128 is not supported on this target}}
      __float128 badFloat = 40; // this SHOULD  trigger a diagnostic

      // not sure if 'no virtual function' is a _deferred_ diagnostic, but testing anyway.
      S s;
      // expected-error@+1 {{SYCL kernel cannot call a virtual function}}
      s.foo();   

      calledFromKernel(10);
    });
  });


  // --- lambda in specialized function testing ---

  //array A is only used to feed the template 
  const int array_size = 4;
  int A[array_size] = {1, 2, 3, 4};
  setup_sycl_operation(A);

  return 0;
}
