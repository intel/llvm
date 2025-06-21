#include "common.hpp"

#include <sycl/usm.hpp>

static constexpr size_t NUM = 10;

int main(int argc, char *argv[]) {
  assert(argc == 2);

  sycl::queue Q;

  int Failed = CommonLoadCheck(Q.get_context(), argv[1]);

#if defined(SYCLBIN_INPUT_STATE)
  auto KBInput = syclexp::get_kernel_bundle<sycl::bundle_state::input>(
      Q.get_context(), {Q.get_device()}, std::string{argv[1]});
  auto KBExe = sycl::build(KBInput);
#elif defined(SYCLBIN_OBJECT_STATE)
  auto KBObj = syclexp::get_kernel_bundle<sycl::bundle_state::object>(
      Q.get_context(), {Q.get_device()}, std::string{argv[1]});
  auto KBExe = sycl::link(KBObj);
#else // defined(SYCLBIN_EXECUTABLE_STATE)
  auto KBExe = syclexp::get_kernel_bundle<sycl::bundle_state::executable>(
      Q.get_context(), {Q.get_device()}, std::string{argv[1]});
#endif

  // TestKernel1 does not have any requirements, so should be there always.
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

  // TestKernel2 should only be there if the device supports fp64.
  if (Q.get_device().has(sycl::aspect::fp64)) {
    assert(KBExe.ext_oneapi_has_kernel("TestKernel2"));
    sycl::kernel TestKernel2 = KBExe.ext_oneapi_get_kernel("TestKernel2");

    Q.submit([&](sycl::handler &CGH) {
       CGH.set_args(Ptr, int{NUM});
       CGH.single_task(TestKernel2);
     }).wait_and_throw();

    for (int I = 0; I < NUM; I++) {
      if (Ptr[I] != static_cast<int>(static_cast<double>(I) / 2.0)) {
        std::cout << "Result: " << Ptr[I] << " expected " << I << "\n";
        ++Failed;
      }
    }
  } else {
    assert(!KBExe.ext_oneapi_has_kernel("TestKernel2"));
  }

  sycl::free(Ptr, Q);
  return Failed;
}
