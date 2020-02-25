// RUN: %clang_cc1 -I %S/Inputs -fsycl -triple spir64 -fsycl-is-device -verify -fsyntax-only %s
//
// Ensure that the SYCL diagnostics that are typically deferred, correctly emitted.
//

#include <sycl.hpp>

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


//  template used to specialize a function that contains a lambda that should
//  result in a deferred diagnostic being emitted.
//  HOWEVER, this is not working presently.  
//  TODO: re-test after new deferred diagnostic system is merged. 
//        restore the "FIX!!" tests below

template <typename T>
void setup_sycl_operation(const T VA[]) {
    
  cl::sycl::range<1> numOfItems;
  cl::sycl::queue deviceQueue;

  deviceQueue.submit([&](cl::sycl::handler &cgh) {
    cgh.single_task<class AName>([=]() {
      // FIX!!  xpected-error@+1 {{zero-length arrays are not permitted in C++}}
      int OverlookedBadArray[0]; 
                       
      // FIX!!   xpected-error@+1 {{__float128 is not supported on this target}}
      __float128 overlookedBadFloat = 40; 
 
    });
  });
}

int main(int argc, char **argv) {

  // --- direct lambda testing ---
  cl::sycl::range<1> numOfItems;
  cl::sycl::queue deviceQueue;

  
  deviceQueue.submit([&](cl::sycl::handler &cgh) {
  
    cgh.single_task<class AName>([=]() {

      // expected-error@+1 {{zero-length arrays are not permitted in C++}}
      int BadArray[0]; 

      // expected-error@+1 {{__float128 is not supported on this target}}
      __float128 badFloat = 40; // this SHOULD  trigger a diagnostic

      // not sure if 'no virtual function' is a _deferred_ diagnostic, but testing anyway.
      S s;
      // expected-error@+1 {{SYCL kernel cannot call a virtual function}}
      s.foo();   

       // expected-note@+1 {{called by 'operator()'}}
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
