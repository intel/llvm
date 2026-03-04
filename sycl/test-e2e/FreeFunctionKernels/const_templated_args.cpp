// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// The name mangling for free function kernels currently does not work with PTX.
// UNSUPPORTED: cuda, hip
// UNSUPPORTED-INTENDED: Not implemented yet for Nvidia/AMD backends.

#include <iostream>
#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/free_function_queries.hpp>
#include <sycl/kernel_bundle.hpp>
#include <sycl/usm.hpp>
#include <type_traits>

namespace syclext = sycl::ext::oneapi;
namespace syclexp = sycl::ext::oneapi::experimental;

static constexpr size_t NUM = 1024;
static constexpr size_t WGSIZE = 16;

template <typename ValueT>
struct KernelConfig {
  ValueT value;
};

template <typename Config>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void launch_kernel(Config cfg, int *ptr) {
  using ConfigValueT = std::remove_reference_t<decltype(cfg.value)>;
  constexpr bool IsReadOnlyConfig =
      std::is_const_v<Config> || std::is_const_v<ConfigValueT>;
  size_t id = syclext::this_work_item::get_nd_item<1>().get_global_linear_id();
  if constexpr (IsReadOnlyConfig) {
    ptr[id] = cfg.value + static_cast<int>(id);
  } else {
    if (cfg.value < 10) {
      cfg.value = 10;
    }
    ptr[id] = cfg.value + static_cast<int>(id) + 1;
  }
}

template <typename Config>
int test_declarations(sycl::queue &q, sycl::context &ctxt, Config &cfg) {
  int *ptr = sycl::malloc_shared<int>(NUM, q);
  auto exe_bndl =
      syclexp::get_kernel_bundle<&launch_kernel<Config>,
                                 sycl::bundle_state::executable>(ctxt);
  sycl::kernel k_func =
      exe_bndl.template ext_oneapi_get_kernel<&launch_kernel<Config>>();

  q.submit([&](sycl::handler &cgh) {
    cgh.set_args(cfg, ptr);
    sycl::nd_range ndr{{NUM}, {WGSIZE}};
    cgh.parallel_for(ndr, k_func);
  }).wait();

  int expected0 = 0;
  using ConfigValueT = std::remove_reference_t<decltype(cfg.value)>;
  constexpr bool IsReadOnlyConfig =
      std::is_const_v<Config> || std::is_const_v<ConfigValueT>;
  if constexpr (IsReadOnlyConfig) {
    expected0 = cfg.value;
  } else {
    expected0 = (cfg.value < 10 ? 10 : cfg.value) + 1;
  }

  int ret = (ptr[0] == expected0) ? 0 : 1;
  sycl::free(ptr, q);
  return ret;
}

int main() {
  sycl::queue q;
  sycl::context ctxt = q.get_context();

  KernelConfig<int> cfg{42};
  const KernelConfig<int> cfg_ro{7};

  const int testInt = 5;
  KernelConfig<decltype(testInt)> cfg_ro1{5};

  static_assert(std::is_const_v<decltype(cfg_ro)>);
  int rc = 0;
  rc |= test_declarations<decltype(cfg)>(q, ctxt, cfg);
  rc |= test_declarations<decltype(cfg_ro)>(q, ctxt, cfg_ro);
  rc |= test_declarations<decltype(cfg_ro1)>(q, ctxt, cfg_ro1);

  return rc;
}
