// REQUIRES: aspect-usm_device_allocations
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/enqueue_functions.hpp>
#include <sycl/ext/oneapi/free_function_queries.hpp>
#include <sycl/kernel_bundle.hpp>
#include <sycl/usm.hpp>

namespace syclext = sycl::ext::oneapi;
namespace syclexp = sycl::ext::oneapi::experimental;

// Struct with static member method
template <typename scalar_t> struct DistsZero {
  static void inc(scalar_t &agg, const scalar_t diff) { agg += diff != 0.0f; }
};

// Free Function Kernel
template <typename scalar_t, typename F>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void test_free_function_kernel(scalar_t *data) {
  auto item = syclext::this_work_item::get_nd_item<1>();
  scalar_t agg = 0.0f;
  scalar_t diff = data[item.get_global_linear_id()];
  F::inc(agg, diff);
  data[item.get_global_linear_id()] = agg;
}

template <auto *kptr, typename... Kargs>
static inline void sycl_kernel_submit(int64_t global_range, int64_t local_range,
                                      ::sycl::queue q, int slm_sz,
                                      Kargs... args) {
  sycl::context ctxt = q.get_context();
  auto exe_bndl =
      syclexp::get_kernel_bundle<kptr, sycl::bundle_state::executable>(ctxt);
  sycl::kernel ker = exe_bndl.template ext_oneapi_get_kernel<kptr>();
  if (slm_sz != 0) {
    syclexp::launch_config cfg{
        ::sycl::nd_range<1>(::sycl::range<1>(global_range),
                            ::sycl::range<1>(local_range)),
        syclexp::properties{syclexp::work_group_scratch_size(slm_sz)}};
    syclexp::nd_launch(q, cfg, ker, args...);
  } else {
    syclexp::launch_config cfg{::sycl::nd_range<1>(
        ::sycl::range<1>(global_range), ::sycl::range<1>(local_range))};
    syclexp::nd_launch(q, cfg, ker, args...);
  }
}

int main() {
  sycl::queue queue;

  constexpr size_t N = 256;
  float *data = sycl::malloc_device<float>(N, queue);

  int64_t global_size = N;
  int64_t local_size = 64;

  sycl_kernel_submit<test_free_function_kernel<float, DistsZero<float>>>(
      global_size, local_size, queue, 0, data);

  queue.wait();
  sycl::free(data, queue);
  return 0;
}
