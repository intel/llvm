// REQUIRES: aspect-usm_shared_allocations
// RUN: %{build} %cxx_std_optionc++20 -o %t.out
// RUN: %{run} %t.out

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/free_function_queries.hpp>
#include <sycl/kernel_bundle.hpp>
#include <sycl/usm.hpp>
#include <type_traits>

namespace syclext = sycl::ext::oneapi;
namespace syclexp = sycl::ext::oneapi::experimental;

static constexpr size_t NUM = 1024;
static constexpr size_t WGSIZE = 16;

inline void kernel_code(float start, float *ptr) {
  size_t id = syclext::this_work_item::get_nd_item<1>().get_global_linear_id();
  ptr[id] = start + static_cast<float>(id);
}

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::work_group_size<WGSIZE>))
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void func_range_wg_1dsize_before(float start, float *ptr) {
  kernel_code(start, ptr);
}


SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::work_group_size<WGSIZE>))
void func_range_wg_1dsize_after(float start, float *ptr) {
  kernel_code(start, ptr);
}

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::work_group_size_hint<WGSIZE>))
void func_range_wg_1dsize_hint_after(float start, float *ptr) {
  kernel_code(start, ptr);
}

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::work_group_size_hint<WGSIZE>))
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void func_range_wg_1dsize_hint_before(float start, float *ptr) {
  kernel_code(start, ptr);
}

template <typename T>
bool check_result(T *ptr) {
  for (size_t i = 0; i < NUM; ++i) {
    const float expected = static_cast<T>(3.14f) + static_cast<T>(i);
    if (ptr[i] != expected)
       return true;
  }
  return false;
}

template <typename T>
static bool call_kernel_code(sycl::queue &q, sycl::kernel &kernel) {
  T *ptr = sycl::malloc_shared<T>(NUM, q);
  q.submit([&](sycl::handler &cgh) {
     cgh.set_args(3.14f, ptr);
     sycl::nd_range ndr{{NUM}, {WGSIZE}};
     cgh.parallel_for(ndr, kernel);
   }).wait();
  const bool ret = check_result<T>(ptr);
  sycl::free(ptr, q);
  return ret;
}

template<typename T, auto* Func>
bool test_function(sycl::queue &q, sycl::context &ctxt) {
  auto exe_bndl =
      syclexp::get_kernel_bundle<Func, sycl::bundle_state::executable>(ctxt);
  sycl::kernel k_func = exe_bndl.template ext_oneapi_get_kernel<Func>();
  bool ret = call_kernel_code<T>(q, k_func);
  auto attrs_info_kernel = sycl::ext::oneapi::experimental::get_kernel_info<Func, sycl::info::kernel::attributes>(ctxt);
}

int main() {
  sycl::queue q;
  sycl::context ctxt = q.get_context();
  
  bool ret = 0;
  ret |= test_function<float, func_range_wg_1dsize_before>(q, ctxt);
  ret |= test_function<float, func_range_wg_1dsize_after>(q, ctxt);
  ret |= test_function<float, func_range_wg_1dsize_hint_before>(q, ctxt);
  ret |= test_function<float, func_range_wg_1dsize_hint_after>(q, ctxt);
  return ret;
}
