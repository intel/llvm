#include "common.hpp"

#include <sycl/usm.hpp>

// TODO: remove SYCL_EXTERNAL from the kernel once it is no longer needed.
auto constexpr SYCLSource = R"===(
#include <sycl/sycl.hpp>

SYCL_EXTERNAL void TestFunc(int *Ptr, int Size);

// use extern "C" to avoid name mangling
extern "C" SYCL_EXTERNAL SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(
    (sycl::ext::oneapi::experimental::single_task_kernel))
void TestKernel1(int *Ptr, int Size) {
  TestFunc(Ptr, Size);
}

)===";

static constexpr size_t NUM = 10;

int main(int argc, char *argv[]) {
  assert(argc == 2);

  sycl::queue Q;

  if (!Q.get_device().ext_oneapi_can_compile(syclexp::source_language::sycl)) {
    std::cout << "Device does not support one of the source languages: "
              << Q.get_device().get_info<sycl::info::device::name>()
              << std::endl;
    return 0;
  }

  int Failed = CommonLoadCheck(Q.get_context(), argv[1]);

  // Load SYCLBIN and compile it.
#if defined(SYCLBIN_INPUT_STATE)
  auto KBInput = syclexp::get_kernel_bundle<sycl::bundle_state::input>(
      Q.get_context(), {Q.get_device()}, std::string{argv[1]});
  auto KBSYCLBINObj = sycl::compile(KBInput);
#elif defined(SYCLBIN_OBJECT_STATE)
  auto KBSYCLBINObj = syclexp::get_kernel_bundle<sycl::bundle_state::object>(
      Q.get_context(), std::string{argv[1]});
#else // defined(SYCLBIN_EXECUTABLE_STATE)
#error "Test does not work with executable state."
#endif

  // Compile source kernel bundle.
  auto KBSrc = syclexp::create_kernel_bundle_from_source(
      Q.get_context(), syclexp::source_language::sycl, SYCLSource);
  syclexp::properties BuildOpts{
      syclexp::build_options{"-fsycl-allow-device-image-dependencies"}};
  auto KBSrcObj = syclexp::compile(KBSrc, BuildOpts);

  // Link the bundles.
  auto KBExe = sycl::link({KBSYCLBINObj, KBSrcObj});

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

  sycl::free(Ptr, Q);
  return Failed;
}
