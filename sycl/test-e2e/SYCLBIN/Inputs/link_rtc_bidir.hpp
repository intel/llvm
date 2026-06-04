#include "common.hpp"

#include <sycl/usm.hpp>

// Cross-origin link regression: the SYCLBIN object provides a kernel
// (TestKernelBin) that imports a SYCL_EXTERNAL function (TestFunc) defined
// in a runtime-compiled SYCL source bundle. After sycl::link({RTC_obj,
// SYCLBIN_obj}) both the RTC-origin kernel (TestKernelRTC) and the
// SYCLBIN-origin kernel (TestKernelBin) must be reachable via
// ext_oneapi_get_kernel and produce correct results.
auto constexpr SYCLSource = R"===(
#include <sycl/ext/oneapi/free_function_kernel_properties.hpp>
#include <sycl/khr/work_item_queries.hpp>

namespace syclkhr = sycl::khr;
namespace syclexp = sycl::ext::oneapi::experimental;

SYCL_EXTERNAL void TestFunc(int *Ptr, int Size) {
  size_t I = syclkhr::this_nd_item<1>().get_global_linear_id();
  if (static_cast<int>(I) < Size)
    Ptr[I] = static_cast<int>(I);
}

extern "C" SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(
    (syclexp::nd_range_kernel<1>))
void TestKernelRTC(int *Ptr, int Size) {
  TestFunc(Ptr, Size);
}

)===";

static constexpr size_t NUM = 32;
static constexpr size_t WGSIZE = 16;

int main(int argc, char *argv[]) {
  assert(argc == 2);

  sycl::queue Q;

  if (!Q.get_device().ext_oneapi_can_compile(syclexp::source_language::sycl)) {
    std::cout << "Device does not support SYCL source kernel compiler: "
              << Q.get_device().get_info<sycl::info::device::name>()
              << std::endl;
    return 0;
  }

  int Failed = CommonLoadCheck(Q.get_context(), argv[1]);

  // Load SYCLBIN object bundle (kernel side).
#if defined(SYCLBIN_INPUT_STATE)
  auto KBInput = syclexp::get_kernel_bundle<sycl::bundle_state::input>(
      Q.get_context(), {Q.get_device()}, std::string{argv[1]});
  auto KBSYCLBINObj = sycl::compile(KBInput);
#elif defined(SYCLBIN_OBJECT_STATE)
  auto KBSYCLBINObj = syclexp::get_kernel_bundle<sycl::bundle_state::object>(
      Q.get_context(), std::string{argv[1]});
#else
#error "Test does not work with executable state."
#endif

  // Compile the RTC source bundle that provides TestFunc and a sibling
  // kernel (TestKernelRTC).
  auto KBSrc = syclexp::create_kernel_bundle_from_source(
      Q.get_context(), syclexp::source_language::sycl, SYCLSource);
  syclexp::properties BuildOpts{
      syclexp::build_options{"-fsycl-allow-device-image-dependencies"}};
  auto KBSrcObj = syclexp::compile(KBSrc, BuildOpts);

  // Link: this is the path that previously merged origins into a single
  // image and routed all lookups through the RTC ProgramManager,
  // making SYCLBIN-origin kernels unreachable.
  auto KBExe = sycl::link({KBSYCLBINObj, KBSrcObj});

  int *Ptr = sycl::malloc_shared<int>(NUM, Q);

  // Sibling RTC-origin kernel: this case already worked before the fix.
  assert(KBExe.ext_oneapi_has_kernel("TestKernelRTC"));
  sycl::kernel KRTC = KBExe.ext_oneapi_get_kernel("TestKernelRTC");
  Q.fill(Ptr, int{-1}, NUM).wait_and_throw();
  Q.submit([&](sycl::handler &CGH) {
     CGH.set_args(Ptr, int{NUM});
     CGH.parallel_for(sycl::nd_range<1>{{NUM}, {WGSIZE}}, KRTC);
   }).wait_and_throw();
  for (size_t I = 0; I < NUM; ++I)
    if (Ptr[I] != static_cast<int>(I)) {
      std::cout << "TestKernelRTC: " << Ptr[I] << " != " << I << "\n";
      ++Failed;
    }

  // SYCLBIN-origin kernel that imports TestFunc from the RTC bundle.
  // This is the regression: ext_oneapi_get_kernel must not return null and
  // the kernel must execute correctly after the cross-origin link.
  assert(KBExe.ext_oneapi_has_kernel("TestKernelBin"));
  sycl::kernel KBin = KBExe.ext_oneapi_get_kernel("TestKernelBin");
  Q.fill(Ptr, int{-1}, NUM).wait_and_throw();
  Q.submit([&](sycl::handler &CGH) {
     CGH.set_args(Ptr, int{NUM});
     CGH.parallel_for(sycl::nd_range<1>{{NUM}, {WGSIZE}}, KBin);
   }).wait_and_throw();
  for (size_t I = 0; I < NUM; ++I)
    if (Ptr[I] != static_cast<int>(I)) {
      std::cout << "TestKernelBin: " << Ptr[I] << " != " << I << "\n";
      ++Failed;
    }

  sycl::free(Ptr, Q);
  return Failed;
}
