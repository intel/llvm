// REQUIRES: aspect-usm_shared_allocations
// RUN: %{build} %cxx_std_optionc++20 -o %t.out
// RUN: %{run} %t.out

// The name mangling for free function kernels currently does not work with PTX.
// UNSUPPORTED: cuda
// UNSUPPORTED-INTENDED: Not implemented yet for Nvidia/AMD backends.

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/free_function_queries.hpp>
#include <sycl/usm.hpp>
#include <sycl/kernel_bundle.hpp>

namespace syclext = sycl::ext::oneapi;
namespace syclexp = sycl::ext::oneapi::experimental;

static constexpr size_t NUM = 1024;
static constexpr size_t WGSIZE = 16;

class TestClass {       
};

struct TestStruct{
};

namespace free_functions::tests{
class TestClass {       
};

struct TestStruct{
};
}

using AliasType = float;

template<typename T>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void tempalted_func(T start, T *ptr) {
  size_t id = syclext::this_work_item::get_nd_item<1>().get_global_linear_id();
}

template<typename T>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void tempalted_func_one_arg(T arg) {
  size_t id = syclext::this_work_item::get_nd_item<1>().get_global_linear_id();
  arg[id] = static_cast<int>(id);
}
/*
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
template<typename Accessor>
void func_accessor(Accessor arg) {
  size_t id = syclext::this_work_item::get_nd_item<1>().get_global_linear_id();
  arg[id] = static_cast<int>(id);
}*/

template<typename T>
static void call_kernel_code(sycl::queue& q, sycl::kernel& kernel) {
  T *ptr = sycl::malloc_shared<T>(NUM, q);
  q.submit([&](sycl::handler &cgh) {
     cgh.set_args(3.14f, ptr);
     sycl::nd_range ndr{{NUM}, {WGSIZE}};
     cgh.parallel_for(ndr, kernel);
   }).wait();
}

static void call_kernel_code_accessor(sycl::queue &q, sycl::kernel &kernel) {
  sycl::buffer<int, 1> buffer(NUM);
  q.submit([&](sycl::handler &cgh) {
     auto accessor = buffer.get_access<sycl::access::mode::write,
                                       sycl::access::target::device>(cgh);
     cgh.set_args(accessor);
     sycl::nd_range ndr{{NUM}, {WGSIZE}};
     cgh.parallel_for(ndr, kernel);
   }).wait();
}

template<typename T>
void test_tempalted_func(sycl::queue& q, sycl::context& ctxt)
{
  // Get a kernel bundle that contains the free function kernel "tempalted_func".
  auto exe_bndl =
      syclexp::get_kernel_bundle<tempalted_func<T>, sycl::bundle_state::executable>(ctxt);
  // Get a kernel object for the "tempalted_func" function from that bundle.
  sycl::kernel k_tempalted_func = exe_bndl.template ext_oneapi_get_kernel<tempalted_func<T>>();
  call_kernel_code<T>(q, k_tempalted_func);
}
/*
void test_accessor_with_target_device(sycl::queue &q, sycl::context &ctxt) {
  // Get a kernel bundle that contains the free function kernel
  // "func_accessor".

  auto exe_bndl =
      syclexp::get_kernel_bundle<func_accessor, sycl::bundle_state::executable>(
          ctxt);
  // Get a kernel object for the "func_accessor" function from that bundle.
  sycl::kernel k_func_accessor =
      exe_bndl.template ext_oneapi_get_kernel<func_accessor>();
  call_kernel_code_accessor(q, k_func_accessor);
}
*/
int main() {
  sycl::queue q;
  sycl::context ctxt = q.get_context();
  test_tempalted_func<TestClass>(q, ctxt);
  test_tempalted_func<TestStruct>(q, ctxt);
  test_tempalted_func<free_functions::tests::TestClass>(q, ctxt);
  test_tempalted_func<free_functions::tests::TestStruct>(q, ctxt);
  test_tempalted_func<AliasType>(q, ctxt);
  //test_accessor_with_target_device(q, ctxt);
  return 0;
}
