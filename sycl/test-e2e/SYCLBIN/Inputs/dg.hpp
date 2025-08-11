#include "common.hpp"

#include <sycl/usm.hpp>

static constexpr size_t NUM = 1024;
static constexpr size_t WGSIZE = 16;
static constexpr float EPS = 0.001;

int main(int argc, char *argv[]) {
  assert(argc == 2);

  sycl::queue Q;
  sycl::device Dev = Q.get_device();

  int Failed = CommonLoadCheck(Q.get_context(), argv[1]);

#if defined(SYCLBIN_INPUT_STATE)
  auto KBInput = syclexp::get_kernel_bundle<sycl::bundle_state::input>(
      Q.get_context(), std::string{argv[1]});
  auto KBExe1 = sycl::build(KBInput);
  auto KBExe2 = sycl::build(KBInput);
#elif defined(SYCLBIN_OBJECT_STATE)
  auto KBObj = syclexp::get_kernel_bundle<sycl::bundle_state::object>(
      Q.get_context(), std::string{argv[1]});
  auto KBExe1 = sycl::link(KBObj);
  auto KBExe2 = sycl::link(KBObj);
#else // defined(SYCLBIN_EXECUTABLE_STATE)
  auto KBExe1 = syclexp::get_kernel_bundle<sycl::bundle_state::executable>(
      Q.get_context(), std::string{argv[1]});
  auto KBExe2 = syclexp::get_kernel_bundle<sycl::bundle_state::executable>(
      Q.get_context(), std::string{argv[1]});
#endif

  sycl::kernel AddK = KBExe1.ext_oneapi_get_kernel("ff_dg_adder");

  // Check presence of device globals.
  assert(KBExe1.ext_oneapi_has_device_global("DG"));
  // Querying a non-existing device global shall not crash.
  assert(!KBExe1.ext_oneapi_has_device_global("bogus_DG"));

  void *DGAddr = KBExe1.ext_oneapi_get_device_global_address("DG", Dev);
  size_t DGSize = KBExe1.ext_oneapi_get_device_global_size("DG");
  assert(DGSize == 4);

  int32_t Val = -1;
  auto CheckVal = [&](int32_t Expected) {
    Val = -1;
    Q.memcpy(&Val, DGAddr, DGSize).wait();
    if (Val != Expected) {
      std::cout << "Val: " << Val << " != " << Expected << '\n';
      ++Failed;
    }
  };

  // Device globals are zero-initialized.
  CheckVal(0);

  // Set the DG.
  Val = 123;
  Q.memcpy(DGAddr, &Val, DGSize).wait();
  CheckVal(123);

  // Run a kernel using it.
  Val = -17;
  Q.submit([&](sycl::handler &CGH) {
     CGH.set_args(Val);
     CGH.single_task(AddK);
   }).wait();
  CheckVal(123 - 17);

  // Test that each bundle has its distinct set of globals.
  DGAddr = KBExe2.ext_oneapi_get_device_global_address("DG", Dev);
  CheckVal(0);

  DGAddr = KBExe1.ext_oneapi_get_device_global_address("DG", Dev);
  CheckVal(123 - 17);

  // Test global with `device_image_scope`. We currently cannot read/write these
  // from the host, but they should work device-only.
  auto SwapK = KBExe2.ext_oneapi_get_kernel("ff_swap");
  int64_t *ValBuf = sycl::malloc_shared<int64_t>(1, Q);
  *ValBuf = -1;
  auto DoSwap = [&]() {
    Q.submit([&](sycl::handler &CGH) {
       CGH.set_args(ValBuf);
       CGH.single_task(SwapK);
     }).wait();
  };

  DoSwap();
  if (*ValBuf != 0) {
    std::cout << "ValBuf: " << *ValBuf << " != 0";
    ++Failed;
  }
  DoSwap();
  if (*ValBuf != -1) {
    std::cout << "ValBuf: " << *ValBuf << " != -1";
    ++Failed;
  }

  sycl::free(ValBuf, Q);

  return Failed;
}
