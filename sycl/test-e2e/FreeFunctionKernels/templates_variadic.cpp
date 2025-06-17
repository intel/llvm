// REQUIRES: aspect-usm_shared_allocations
// RUN: %{build} -DNDEBUG -o %t.out
// RUN: %{run} %t.out

#include <cassert>
#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/free_function_queries.hpp>
#include <sycl/kernel_bundle.hpp>
#include <sycl/usm.hpp>

namespace syclext = sycl::ext::oneapi;
namespace syclexp = sycl::ext::oneapi::experimental;

static constexpr size_t NUM = 1024;
static constexpr size_t WGSIZE = 16;

template <typename T, typename... Args>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void variadic(T *ptr, Args... args) {
  size_t id = syclext::this_work_item::get_nd_item<1>().get_global_linear_id();
  ((ptr[id] = static_cast<T>(5) + args), ...);
}

namespace Templates::Tests {
template <typename T, typename... Args>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void variadic(T *ptr, Args... args) {
  size_t id = syclext::this_work_item::get_nd_item<1>().get_global_linear_id();
  ((ptr[id] = static_cast<T>(5) + args), ...);
}
} // namespace Templates::Tests

template <typename T> void check_result(T *ptr) {
  for (size_t i = 0; i < NUM; ++i) {
    const T expected = static_cast<T>(5) + static_cast<T>(i);
    assert(ptr[i] == expected &&
           "Kernel execution did not produce the expected result");
  }
}

template <typename T>
static void call_kernel_code(sycl::queue &q, sycl::kernel &kernel) {
  T *ptr = sycl::malloc_shared<T>(NUM, q);
  q.submit([&](sycl::handler &cgh) {
     if constexpr (std::is_same<T, float>::value)
       cgh.set_args(ptr, 1.0f);
     else
       cgh.set_args(ptr, 1, 2, 3);
     sycl::nd_range ndr{{NUM}, {WGSIZE}};
     cgh.parallel_for(ndr, kernel);
   }).wait();
  check_result<T>(ptr);
  sycl::free(ptr, q);
}

template <auto Func, typename T, typename... Args>
void test_variadic(sycl::queue &q, sycl::context &ctxt) {
  auto exe_bndl =
      syclexp::get_kernel_bundle<Func, sycl::bundle_state::executable>(ctxt);
  sycl::kernel k_variadic = exe_bndl.template ext_oneapi_get_kernel<Func>();
  call_kernel_code<T>(q, k_variadic);
}

int main() {
  sycl::queue q;
  sycl::context ctxt = q.get_context();

  test_variadic<variadic<float, float>, float, float>(q, ctxt);
  test_variadic<variadic<int, int, int, int>, int, int, int, int>(q, ctxt);
  test_variadic<Templates::Tests::variadic<float, float>, float, float>(q,
                                                                        ctxt);
  test_variadic<Templates::Tests::variadic<int, int, int, int>, int, int, int,
                int>(q, ctxt);
  return 0;
}
