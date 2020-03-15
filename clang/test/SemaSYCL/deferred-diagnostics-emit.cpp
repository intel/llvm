// RUN: %clang_cc1  -fsycl -triple spir64 -fsycl-is-device -verify -fsyntax-only  %s
//
// Ensure that the SYCL diagnostics that are typically deferred are correctly emitted.

// testing that the deferred diagnostics work in conjunction with the SYCL namespaces.
inline namespace cl {
namespace sycl {

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel_single_task(Func kernelFunc) {
  // expected-note@+1 2{{called by 'kernel_single_task<AName, (lambda}}
  kernelFunc();
}

} // namespace sycl
} // namespace cl

//variadic functions from SYCL kernels emit a deferred diagnostic
void variadic(int, ...) {}

int calledFromKernel(int a) {
  // expected-error@+1 {{zero-length arrays are not permitted in C++}}
  int MalArray[0];

  // expected-error@+1 {{__float128 is not supported on this target}}
  __float128 malFloat = 40;

  //expected-error@+1 {{SYCL kernel cannot call a variadic function}}
  variadic(5);

  return a + 20;
}

//  template used to specialize a function that contains a lambda that should
//  result in a deferred diagnostic being emitted.
//  HOWEVER, this is not working presently.
//  TODO: re-test after new deferred diagnostic system is merged.
//        restore the "FIX!!" tests below

template <typename T>
void setup_sycl_operation(const T VA[]) {

  cl::sycl::kernel_single_task<class AName>([]() {
    // FIX!!  xpected-error@+1 {{zero-length arrays are not permitted in C++}}
    int OverlookedBadArray[0];

    // FIX!!   xpected-error@+1 {{__float128 is not supported on this target}}
    __float128 overlookedBadFloat = 40;
  });
}

int main(int argc, char **argv) {

  // --- direct lambda testing ---
  cl::sycl::kernel_single_task<class AName>([]() {
    // expected-error@+1 {{zero-length arrays are not permitted in C++}}
    int BadArray[0];

    // expected-error@+1 {{__float128 is not supported on this target}}
    __float128 badFloat = 40; // this SHOULD  trigger a diagnostic

    //expected-error@+1 {{SYCL kernel cannot call a variadic function}}
    variadic(5);

    // expected-note@+1 {{called by 'operator()'}}
    calledFromKernel(10);
  });

  // --- lambda in specialized function testing ---

  //array A is only used to feed the template
  const int array_size = 4;
  int A[array_size] = {1, 2, 3, 4};
  setup_sycl_operation(A);

  return 0;
}
