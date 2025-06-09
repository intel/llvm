// REQUIRES: aspect-usm_shared_allocations
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include <iostream>
#include <sycl/ext/oneapi/free_function_queries.hpp>
#include <sycl/ext/oneapi/get_kernel_info.hpp>
#include <sycl/kernel_bundle.hpp>
#include <sycl/usm.hpp>
#include <type_traits>

namespace syclext = sycl::ext::oneapi;
namespace syclexp = sycl::ext::oneapi::experimental;

static constexpr size_t NUM = 1024;
static constexpr size_t WGSIZE = 16;

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::work_group_size<WGSIZE>))
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void func(float start, float *ptr) {
  size_t id = syclext::this_work_item::get_nd_item<1>().get_global_linear_id();
  ptr[id] = start + static_cast<float>(id);
}

bool check_result(int *ptr) {
  for (size_t i = 0; i < NUM; ++i) {
    const int expected = 3 + static_cast<int>(i);
    if (ptr[i] != expected)
      return true;
  }
  return false;
}

static bool call_kernel_code(sycl::queue &q, sycl::kernel &kernel) {
  int *ptr = sycl::malloc_shared<int>(NUM, q);
  q.submit([&](sycl::handler &cgh) {
     cgh.set_args(3, ptr);
     sycl::nd_range ndr{{NUM}, {WGSIZE}};
     cgh.parallel_for(ndr, kernel);
   }).wait();
  const bool ret = check_result(ptr);
  sycl::free(ptr, q);
  return ret;
}

bool test_ctxt_dev(sycl::kernel &k, sycl::queue &q) {
  const auto wg_size_cmp =
      k.get_info<sycl::info::kernel_device_specific::work_group_size>(
          q.get_device());
  const auto wg_size = syclexp::get_kernel_info<
      func, sycl::info::kernel_device_specific::work_group_size>(
      q.get_context(), q.get_device());
  if (wg_size_cmp != wg_size)
    std::cerr << "Work group size from get_info: " << wg_size_cmp
              << " is not equal to work group size from get_kernel_info: "
              << wg_size << std::endl;
  return wg_size_cmp == wg_size;
}

bool test_ctxt(sycl::kernel &k, sycl::queue &q) {
  const auto attributes =
      syclexp::get_kernel_info<func, sycl::info::kernel::attributes>(
          q.get_context());
  const std::string wg_size_str = "work_group_size(";
  if (attributes.empty() || attributes.find(wg_size_str) == std::string::npos) {
    std::cerr << "Work group size attribute is not found in kernel attributes"
              << std::endl;
    return false;
  }
  auto wg_size_pos = attributes.find(wg_size_str);
  wg_size_pos += wg_size_str.size();
  const auto comma_pos = attributes.find(',', wg_size_pos);
  if (comma_pos == std::string::npos) {
    std::cerr << "Comma not found in work group size attribute string"
              << std::endl;
    return false;
  }

  const auto wg_size_str_value =
      attributes.substr(wg_size_pos, comma_pos - wg_size_pos);
  const size_t wg_size = std::stoul(wg_size_str_value);
  if (wg_size != WGSIZE) {
    std::cerr << "Work group size from attributes: " << wg_size
              << " is not equal to expected work group size: " << WGSIZE
              << std::endl;
    return false;
  }
  return true;
}

bool test_queue(sycl::kernel &k, sycl::queue &q) {
  const auto wg_size_cmp =
      k.get_info<sycl::info::kernel_device_specific::work_group_size>(
          q.get_device());
  const auto wg_size = syclexp::get_kernel_info<
      func, sycl::info::kernel_device_specific::work_group_size>(q);
  if (wg_size_cmp != wg_size)
    std::cerr << "Work group size from get_info: " << wg_size_cmp
              << " is not equal to work group size from get_kernel_info: "
              << wg_size << std::endl;
  return wg_size_cmp == wg_size;
}

int main() {
  sycl::queue q;
  sycl::context ctxt = q.get_context();

  auto exe_bndl =
      syclexp::get_kernel_bundle<func, sycl::bundle_state::executable>(ctxt);
  sycl::kernel k_func = exe_bndl.template ext_oneapi_get_kernel<func>();
  call_kernel_code(q, k_func);

  bool ret = test_ctxt_dev(k_func, q);
  ret &= test_ctxt(k_func, q);
  ret &= test_queue(k_func, q);
  return ret ? 0 : 1;
}
