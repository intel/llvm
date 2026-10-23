// REQUIRES: sycl-jit, aspect-usm_shared_allocations

// RUN: %{build} -o %t.out
// RUN: %if hip %{ env SYCL_JIT_AMDGCN_PTX_TARGET_CPU=%{amd_arch} %} %{run} %t.out
// RUN: %if hip %{ env SYCL_JIT_AMDGCN_PTX_TARGET_CPU=%{amd_arch} %} %{run} %t.out keep

// Cached source-built bundles must support execution and cleanup during static
// destruction, after ordinary runtime static destructors would have run.
#include <sycl/detail/core.hpp>
#include <sycl/kernel_bundle.hpp>
#include <sycl/usm.hpp>

#include <cassert>
#include <optional>
#include <vector>

namespace syclex = sycl::ext::oneapi::experimental;
using ExecutableBundle = sycl::kernel_bundle<sycl::bundle_state::executable>;

constexpr const char *Source = R"(
#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/free_function_kernel_properties.hpp>
extern "C" SYCL_EXTERNAL
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((sycl::ext::oneapi::experimental::single_task_kernel))
void increment(int *Value) { ++*Value; }
)";

struct KernelCache {
  std::vector<ExecutableBundle> Bundles;
  std::optional<sycl::queue> Queue;

  ~KernelCache() {
    for (auto &Bundle : Bundles) {
      auto Kernel = Bundle.ext_oneapi_get_kernel("increment");
      assert(Kernel.get_info<sycl::info::kernel::num_args>() == 1);
      int *Value = sycl::malloc_shared<int>(1, *Queue);
      *Value = 0;
      Queue->submit([&](sycl::handler &CGH) {
        CGH.set_arg(0, Value);
        CGH.single_task(Kernel);
      });
      Queue->wait_and_throw();
      assert(*Value == 1);
      sycl::free(Value, *Queue);
    }
  }
};

int main(int Argc, char **) {
  sycl::queue Q;
  static KernelCache Cache;
  assert(Q.get_device().ext_oneapi_can_build(syclex::source_language::sycl));
  auto SourceBundle = syclex::create_kernel_bundle_from_source(
      Q.get_context(), syclex::source_language::sycl, Source);
  for (int I = 0; I < 2; ++I) {
    auto Bundle = syclex::build(SourceBundle);
    assert(Bundle.ext_oneapi_get_kernel("increment")
               .get_info<sycl::info::kernel::num_args>() == 1);
    if (Argc > 1) {
      Cache.Queue = Q;
      Cache.Bundles.push_back(Bundle);
    }
  }
}
