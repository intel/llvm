// REQUIRES: aspect-usm_shared_allocations
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// UNSUPPORTED: cuda, hip
// UNSUPPORTED-INTENDED: Device incompatible error

// XFAIL: target-native_cpu
// XFAIL-TRACKER: https://github.com/intel/llvm/issues/20142

#include <iostream>
#include <sycl/ext/oneapi/free_function_queries.hpp>
#include <sycl/ext/oneapi/get_kernel_info.hpp>
#include <sycl/kernel_bundle.hpp>

#include <sycl/usm.hpp>

namespace syclext = sycl::ext::oneapi;
namespace syclexp = sycl::ext::oneapi::experimental;

static constexpr size_t NUM = 1024;
static constexpr size_t WGSIZE = 32;
static constexpr size_t SGSIZE = 16;

inline void kernel_code(float start, float *ptr) {
  size_t id = syclext::this_work_item::get_nd_item<1>().get_global_linear_id();
  ptr[id] = start + static_cast<float>(id);
}

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::work_group_size<WGSIZE>))
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void range_wg_1dsize_before(float start, float *ptr) {
  kernel_code(start, ptr);
}

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::work_group_size<WGSIZE>))
void range_wg_1dsize_after(float start, float *ptr) { kernel_code(start, ptr); }

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::work_group_size_hint<WGSIZE>))
void range_wg_1dsize_hint_after(float start, float *ptr) {
  kernel_code(start, ptr);
}

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::work_group_size_hint<WGSIZE>))
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void range_wg_1dsize_hint_before(float start, float *ptr) {
  kernel_code(start, ptr);
}

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::sub_group_size<SGSIZE>))
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void range_sg_1dsize_before(float start, float *ptr) {
  kernel_code(start, ptr);
}

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::sub_group_size<SGSIZE>))
void range_sg_1dsize_after(float start, float *ptr) { kernel_code(start, ptr); }

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::device_has<sycl::aspect::gpu>))
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void range_has_before(float start, float *ptr) { kernel_code(start, ptr); }

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::device_has<sycl::aspect::gpu>))
void range_has_after(float start, float *ptr) { kernel_code(start, ptr); }

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::work_group_size<WGSIZE>))
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::sub_group_size<SGSIZE>))
void range_several_after(float start, float *ptr) { kernel_code(start, ptr); }

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::work_group_size<WGSIZE>))
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::sub_group_size<SGSIZE>))
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void range_several_before(float start, float *ptr) { kernel_code(start, ptr); }

template <typename T> bool check_result(T *ptr) {
  for (size_t i = 0; i < NUM; ++i) {
    const T expected = static_cast<T>(3.14f) + static_cast<T>(i);
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

template <auto *Func, typename T, typename GetKernelInfoParam>
bool test(sycl::queue &q, sycl::context &ctxt, std::string_view name) {
  auto exe_bndl =
      syclexp::get_kernel_bundle<Func, sycl::bundle_state::executable>(ctxt);
  sycl::kernel k_func = exe_bndl.template ext_oneapi_get_kernel<Func>();
  bool ret = call_kernel_code<T>(q, k_func);
  auto value = syclexp::get_kernel_info<Func, GetKernelInfoParam>(q);
  const auto kernel_ids = exe_bndl.get_kernel_ids();
  if (kernel_ids.empty())
    return true;
  sycl::kernel k = exe_bndl.get_kernel(kernel_ids[0]);
  const size_t kernel_value = k.get_info<GetKernelInfoParam>(q.get_device());
  ret |= (value != kernel_value);
  if (ret)
    std::cout << "Test " << name
              << " did not pass: value got from get_kernel_info " << value
              << ", value got from kernel get_info " << kernel_value
              << std::endl;
  return ret;
}

template <auto *Func, typename T>
bool test_has_desc(sycl::queue &q, sycl::context &ctxt) {
  auto exe_bndl =
      syclexp::get_kernel_bundle<Func, sycl::bundle_state::executable>(ctxt);
  sycl::kernel k_func = exe_bndl.template ext_oneapi_get_kernel<Func>();
  return call_kernel_code<T>(q, k_func);
}

using wg_size_desc = sycl::info::kernel_device_specific::work_group_size;
using sg_size_desc = sycl::info::kernel_device_specific::compile_sub_group_size;

template <auto *Func, typename T>
bool test_several_properties(sycl::queue &q, sycl::context &ctxt,
                             std::string_view name) {
  auto exe_bndl =
      syclexp::get_kernel_bundle<Func, sycl::bundle_state::executable>(ctxt);
  sycl::kernel k_func = exe_bndl.template ext_oneapi_get_kernel<Func>();
  bool ret = call_kernel_code<T>(q, k_func);
  const size_t value_wg_size = syclexp::get_kernel_info<Func, wg_size_desc>(q);
  const size_t value_sg_size = syclexp::get_kernel_info<Func, sg_size_desc>(q);
  const auto kernel_ids = exe_bndl.get_kernel_ids();
  if (kernel_ids.empty())
    return true;
  sycl::kernel k = exe_bndl.get_kernel(kernel_ids[0]);
  const size_t kernel_value_wg = k.get_info<wg_size_desc>(q.get_device());
  const size_t kernel_value_sg = k.get_info<sg_size_desc>(q.get_device());
  ret |= (value_wg_size != kernel_value_wg);
  ret |= (value_sg_size != kernel_value_sg);
  if (ret)
    std::cout << "Test " << name << " did not pass: value_wg_size "
              << value_wg_size << ", value_sg_size " << value_sg_size
              << ", kernel_value_wg " << kernel_value_wg << ", kernel_value_sg "
              << kernel_value_sg << std::endl;
  return ret;
}

int main() {
  sycl::queue q;
  sycl::context ctxt = q.get_context();

  int ret = 0;
  ret |= test<range_wg_1dsize_before, float, wg_size_desc>(
      q, ctxt, "range_wg_1dsize_before");
  ret |= test<range_wg_1dsize_after, float, wg_size_desc>(
      q, ctxt, "range_wg_1dsize_after");
  ret |= test<range_wg_1dsize_hint_before, float, wg_size_desc>(
      q, ctxt, "range_wg_1dsize_hint_before");
  ret |= test<range_wg_1dsize_hint_after, float, wg_size_desc>(
      q, ctxt, "range_wg_1dsize_hint_after");
  ret |= test<range_sg_1dsize_before, float, sg_size_desc>(
      q, ctxt, "range_sg_1dsize_before");
  ret |= test<range_sg_1dsize_after, float, sg_size_desc>(
      q, ctxt, "range_sg_1dsize_after");
  ret |= test_has_desc<range_has_before, float>(q, ctxt);
  ret |= test_has_desc<range_has_after, float>(q, ctxt);
  ret |= test_several_properties<range_several_before, float>(
      q, ctxt, "range_several_before");
  ret |= test_several_properties<range_several_after, float>(
      q, ctxt, "range_several_after");
  return ret;
}
