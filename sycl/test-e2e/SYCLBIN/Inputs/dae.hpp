#include "common.hpp"

#include <sycl/usm.hpp>

static constexpr size_t NUM = 1024;
static constexpr size_t WGSIZE = 16;
static constexpr float EPS = 0.001;

int main(int argc, char *argv[]) {
  assert(argc == 2);

  sycl::queue Q;

  int Failed = CommonLoadCheck(Q.get_context(), argv[1]);

#if defined(SYCLBIN_INPUT_STATE)
  auto KBInput = syclexp::get_kernel_bundle<sycl::bundle_state::input>(
      Q.get_context(), std::string{argv[1]});
  auto KBExe = sycl::build(KBInput);
#elif defined(SYCLBIN_OBJECT_STATE)
  auto KBObj = syclexp::get_kernel_bundle<sycl::bundle_state::object>(
      Q.get_context(), std::string{argv[1]});
  auto KBExe = sycl::link(KBObj);
#else // defined(SYCLBIN_EXECUTABLE_STATE)
  auto KBExe = syclexp::get_kernel_bundle<sycl::bundle_state::executable>(
      Q.get_context(), std::string{argv[1]});
#endif

  assert(KBExe.ext_oneapi_has_kernel("iota"));
  sycl::kernel IotaKern = KBExe.ext_oneapi_get_kernel("iota");

  float *Ptr = sycl::malloc_shared<float>(NUM, Q);
  Q.submit([&](sycl::handler &CGH) {
     // First arugment is unused, but should still be passed, even if eliminated
     // by DAE.
     CGH.set_args(3.14f, Ptr);
     CGH.parallel_for(sycl::nd_range{{NUM}, {WGSIZE}}, IotaKern);
   }).wait_and_throw();

  for (int I = 0; I < NUM; I++) {
    const float Truth = static_cast<float>(I);
    if (std::abs(Ptr[I] - Truth) > EPS) {
      std::cout << "Result: " << Ptr[I] << " expected " << I << "\n";
      ++Failed;
    }
  }
  sycl::free(Ptr, Q);
  return Failed;
}
