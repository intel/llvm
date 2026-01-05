// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

/*
 * Test to check class/struct type with virtual methods as SYCL free function
 * kernel arguments.
 */

#include <cmath>
#include <iostream>
#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/free_function_queries.hpp>
#include <sycl/kernel_bundle.hpp>
#include <sycl/usm.hpp>

namespace syclext = sycl::ext::oneapi;
namespace syclexp = sycl::ext::oneapi::experimental;

static constexpr size_t NUM = 1024;
static constexpr size_t WGSIZE = 16;
static constexpr auto FFTestMark = "Free function Kernel Test:";
static constexpr float offset = 1.1f;

class Base {
public:
  virtual void virtual_method(float start) = 0;
  virtual ~Base() = default;
};

class TestClass : public Base {
  float data = 0.0f;

public:
  void virtual_method(float start) override {}

  float calculate(float start, size_t id) {
    return start + static_cast<float>(id) + data;
  }

  void setData(float value) { data = value; }
};

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<2>))
void func_range(TestClass *acc, float *ptr) {
  size_t id = syclext::this_work_item::get_nd_item<1>().get_global_linear_id();
  ptr[id] = acc->calculate(3.14f, id);
}

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::single_task_kernel))
void func_single(TestClass *acc, float *ptr) {
  size_t id = syclext::this_work_item::get_nd_item<1>().get_global_linear_id();
  ptr[id] = acc->calculate(3.14f, id);
}

int check_result(float *ptr) {
  constexpr float diff_cmp = 1e-4f;
  for (size_t i = 0; i < NUM; ++i) {
    const float expected = 3.14f + static_cast<float>(i) + offset;
    if (std::fabs(ptr[i] - expected) > diff_cmp)
      return 1;
  }
  return 0;
}

int call_kernel_code(sycl::queue &q, sycl::kernel &kernel) {
  float *ptr = sycl::malloc_shared<float>(NUM, q);
  TestClass *obj = sycl::malloc_shared<TestClass>(1, q);
  obj->setData(offset);

  q.submit([&](sycl::handler &cgh) {
     cgh.set_args(obj, ptr);
     sycl::nd_range ndr{{NUM}, {WGSIZE}};
     cgh.parallel_for(ndr, kernel);
   }).wait();
  int ret = check_result(ptr);
  sycl::free(ptr, q);
  sycl::free(obj, q);
  return ret;
}

template <auto *Func>
int test_arg_with_virtual_method(sycl::queue &q, sycl::context &ctxt,
                                 std::string_view name) {
  auto exe_bndl =
      syclexp::get_kernel_bundle<Func, sycl::bundle_state::executable>(ctxt);
  sycl::kernel k_func = exe_bndl.template ext_oneapi_get_kernel<Func>();
  int ret = call_kernel_code(q, k_func);
  if (ret != 0)
    std::cerr << FFTestMark << name << " failed\n";
  return ret;
}

int main() {
  sycl::queue q;
  sycl::context ctxt = q.get_context();
  sycl::device dev = q.get_device();

  int ret =
      test_arg_with_virtual_method<func_range>(q, ctxt, "virtual_method_range");
  ret |= test_arg_with_virtual_method<func_single>(q, ctxt,
                                                   "virtual_method_single");
  return ret;
}
