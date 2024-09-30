// REQUIRES: gpu, level_zero

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// Verifies that a kernel can be launched with properities.

#include <numeric>
#include <sycl/detail/core.hpp>
#include <sycl/properties/all_properties.hpp>
#include <sycl/usm.hpp>

using namespace sycl;
using namespace sycl::ext::intel::experimental;
using namespace sycl::ext::oneapi::experimental;

// TODO: remove SYCL_EXTERNAL once it is no longer needed.
auto constexpr SYCLSource = R"===(
#include <sycl/sycl.hpp>

// use extern "C" to avoid name mangling
extern "C" SYCL_EXTERNAL SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((sycl::ext::oneapi::experimental::nd_range_kernel<1>))
void ff_cp() {}
)===";

int main() {
  namespace syclex = sycl::ext::oneapi::experimental;
  using source_kb = sycl::kernel_bundle<sycl::bundle_state::ext_oneapi_source>;
  using exe_kb = sycl::kernel_bundle<sycl::bundle_state::executable>;

  queue Queue{};
  sycl::context Ctx = Queue.get_context();

  syclex::properties Properties{};

  // Create from source.
  source_kb kbSrc = syclex::create_kernel_bundle_from_source(
      Ctx, syclex::source_language::sycl, SYCLSource, syclex::properties{});

  // Compilation of empty prop list, no devices.
  exe_kb kbExe1 = syclex::build(kbSrc);

  // clang-format off

  sycl::nd_range<1> R1{{10}, {1}};
  // extern "C" was used, so the name "ff_cp" is not mangled and can be used directly.
  sycl::kernel Kernel = kbExe1.ext_oneapi_get_kernel("ff_cp");

  // clang-format on

  Queue.submit([&](sycl::handler &Handler) {
    Handler.parallel_for(R1, Properties, Kernel);
  });
  Queue.wait();

  return 0;
}
