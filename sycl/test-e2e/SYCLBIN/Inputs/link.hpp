#include "common.hpp"

#include <sycl/usm.hpp>

namespace syclex = sycl::ext::oneapi::experimental;

#ifdef SYCLBIN_USE_FAST_LINK
static constexpr bool USE_FAST_LINK = true;
#else
static constexpr bool USE_FAST_LINK = false;
#endif

static constexpr size_t NUM = 10;

int main(int argc, char *argv[]) {
  assert(argc == 3);

  sycl::queue Q;

  int Failed = CommonLoadCheck(Q.get_context(), argv[1]) +
               CommonLoadCheck(Q.get_context(), argv[2]);

  // Load SYCLBINs.
#if defined(SYCLBIN_INPUT_STATE)
  auto KBInput1 = syclex::get_kernel_bundle<sycl::bundle_state::input>(
      Q.get_context(), {Q.get_device()}, std::string{argv[1]});
  auto KBInput2 = syclex::get_kernel_bundle<sycl::bundle_state::input>(
      Q.get_context(), {Q.get_device()}, std::string{argv[2]});

  // Compile the bundles.
  auto KBObj1 = sycl::compile(KBInput1);
  auto KBObj1Extra = sycl::compile(KBInput1);
  auto KBObj2 = sycl::compile(KBInput2);
  auto KBObj2Extra = sycl::compile(KBInput2);
#elif defined(SYCLBIN_OBJECT_STATE)
  auto KBObj1 = syclex::get_kernel_bundle<sycl::bundle_state::object>(
      Q.get_context(), {Q.get_device()}, std::string{argv[1]});
  auto KBObj1Extra = syclex::get_kernel_bundle<sycl::bundle_state::object>(
      Q.get_context(), {Q.get_device()}, std::string{argv[1]});
  auto KBObj2 = syclex::get_kernel_bundle<sycl::bundle_state::object>(
      Q.get_context(), {Q.get_device()}, std::string{argv[2]});
  auto KBObj2Extra = syclex::get_kernel_bundle<sycl::bundle_state::object>(
      Q.get_context(), {Q.get_device()}, std::string{argv[2]});
#else // defined(SYCLBIN_EXECUTABLE_STATE)
#error "Test does not work with executable state."
#endif

  // Check that linking without all symbols resolved will result in an
  // exception.
  try {
    auto KBExeFail = syclexp::link(
        {KBObj2}, syclexp::properties{syclexp::fast_link{USE_FAST_LINK}});
    std::cout << "No exception thrown for only KBObj2" << std::endl;
    ++Failed;
  } catch (sycl::exception &E) {
    if (E.code() != sycl::make_error_code(sycl::errc::invalid)) {
      std::cout << "Unexpected exception code for only KBObj2" << std::endl;
      ++Failed;
    }
  } catch (...) {
    std::cout << "Caught unexpected exception for only KBObj2" << std::endl;
    ++Failed;
  }

  // Check that linking bundles with conflicting kernels will result in an
  // exception.
  try {
    auto KBExeFail =
        syclexp::link({KBObj1, KBObj1Extra, KBObj2},
                      syclexp::properties{syclexp::fast_link{USE_FAST_LINK}});
    std::cout << "No exception thrown for double KBObj1" << std::endl;
    ++Failed;
  } catch (sycl::exception &E) {
    if (E.code() != sycl::make_error_code(sycl::errc::invalid)) {
      std::cout << "Unexpected exception code for double KBObj1" << std::endl;
      ++Failed;
    }
  } catch (...) {
    std::cout << "Caught unexpected exception for double KBObj1" << std::endl;
    ++Failed;
  }

  // Check that linking bundles with conflicting exported symbols will result in
  // an exception.
  try {
    auto KBExeFail =
        syclexp::link({KBObj1, KBObj2, KBObj2Extra},
                      syclexp::properties{syclexp::fast_link{USE_FAST_LINK}});
    std::cout << "No exception thrown for double KBObj2" << std::endl;
    ++Failed;
  } catch (sycl::exception &E) {
    if (E.code() != sycl::make_error_code(sycl::errc::invalid)) {
      std::cout << "Unexpected exception code for double KBObj2" << std::endl;
      ++Failed;
    }
  } catch (...) {
    std::cout << "Caught unexpected exception for double KBObj2" << std::endl;
    ++Failed;
  }

  // Link the bundles.
  auto KBExe = syclexp::link(
      {KBObj1, KBObj2}, syclexp::properties{syclexp::fast_link{USE_FAST_LINK}});

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
