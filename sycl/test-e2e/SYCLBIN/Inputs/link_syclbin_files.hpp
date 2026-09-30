#include "common.hpp"

#include <sycl/usm.hpp>

static constexpr size_t NUM = 10;

// Loads an executable-state SYCLBIN produced by linking SYCLBIN files with
// -fsycl-link and runs the kernel from Inputs/importing_kernel.cpp, which calls
// the SYCL_EXTERNAL function from Inputs/exporting_function.cpp.
int main(int argc, char *argv[]) {
  assert(argc == 2);

  sycl::queue Q;

  int Failed = CommonLoadCheck(Q.get_context(), argv[1]);

  auto KBExe = syclexp::get_kernel_bundle<sycl::bundle_state::executable>(
      Q.get_context(), {Q.get_device()}, std::string{argv[1]});

  assert(KBExe.ext_oneapi_has_kernel("TestKernel1"));
  sycl::kernel TestKernel1 = KBExe.ext_oneapi_get_kernel("TestKernel1");

  int *Ptr = sycl::malloc_shared<int>(NUM, Q);
  Q.fill(Ptr, int{0}, NUM).wait_and_throw();

  Q.submit([&](sycl::handler &CGH) {
     CGH.set_args(Ptr, int{NUM});
     CGH.single_task(TestKernel1);
   }).wait_and_throw();

  for (int I = 0; I < NUM; I++) {
    if (Ptr[I] != I) {
      std::cout << "Result: " << Ptr[I] << " expected " << I << "\n";
      ++Failed;
    }
  }

  sycl::free(Ptr, Q);
  return Failed;
}
