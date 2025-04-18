// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// This test checks whether traits for free function kernels works for free
// function kernels which are decorated with nd_range_kernel or
// single_task_kernel properties.

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/free_function_traits.hpp>
#include <sycl/ext/oneapi/free_function_queries.hpp>

namespace syclext = sycl::ext::oneapi;
namespace syclexp = sycl::ext::oneapi::experimental;

template <bool ResultValue, bool ExpectedResult>
int performCheck(std::string_view ErrorMessage) {
  if (ResultValue != ExpectedResult) {
    std::cerr << ErrorMessage << std::endl;
    return 1;
  }
  return 0;
}

void func(int *Ptr) {}

template <typename T> void templatedFunc(T *Ptr) {}

class ClassScope {
public:
  static void staticMemberFunc(int *Ptr) {}
};

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<2>))
void ndRangeFreeFuncKernel(int *Ptr) {
  size_t Item =
      syclext::this_work_item::get_nd_item<2>().get_global_linear_id();
  Ptr[Item] = static_cast<int>(Item);
}

namespace ns {
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::single_task_kernel))
void nsSingleTaskFreeFuncKernel(int *Ptr, size_t NumOfElements) {
  for (size_t i = 0; i < NumOfElements; ++i) {
    Ptr[i] = static_cast<int>(i);
  }
}

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<3>))
void nsNdRangeFreeFuncKernel(float *Ptr) {
  size_t Item =
      syclext::this_work_item::get_nd_item<3>().get_global_linear_id();
  Ptr[Item] = static_cast<float>(Item);
}
} // namespace ns

int main() {
  int Failed = 0;

  // FIXME: need to add checks for free function which are static member
  // functions decorated with single_task_kernel and nd_range_kernel properties
  // when that feature will be implemented.

  // FIXME: need to add checks for templated free function kernels that are
  // decorated with single_task_kernel and nd_range_kernel properties when that
  // feature will be implemented.

  Failed += performCheck<syclexp::is_nd_range_kernel_v<func, 1>, false>(
      "Expected false because func() is not a nd_range free function kernel.");
  Failed += performCheck<syclexp::is_single_task_kernel_v<func>, false>(
      "Expected false because func() is not a single task free function "
      "kernel.");
  Failed += performCheck<syclexp::is_kernel_v<func>, false>(
      "Expected false because func() is not a free function kernel.");

  Failed += performCheck<syclexp::is_nd_range_kernel_v<templatedFunc<float>, 1>,
                         false>("Expected false because templatedFunc() is not "
                                "a nd_range free function kernel.");
  Failed +=
      performCheck<syclexp::is_single_task_kernel_v<templatedFunc<double>>,
                   false>("Expected false because templatedFunc() is not a "
                          "single task free function kernel.");
  Failed += performCheck<syclexp::is_kernel_v<templatedFunc<int>>, false>(
      "Expected false because templatedFunc() is not a free function kernel.");

  Failed += performCheck<
      syclexp::is_nd_range_kernel_v<ClassScope::staticMemberFunc, 1>, false>(
      "Expected false because ClassScope::staticMemberFunc() is not a nd_range "
      "free function kernel.");
  Failed += performCheck<
      syclexp::is_single_task_kernel_v<ClassScope::staticMemberFunc>, false>(
      "Expected false because ClassScope::staticMemberFunc() is not a single "
      "task free function kernel.");
  Failed +=
      performCheck<syclexp::is_kernel_v<ClassScope::staticMemberFunc>, false>(
          "Expected false because ClassScope::staticMemberFunc() is not a free "
          "function kernel.");

  Failed +=
      performCheck<syclexp::is_nd_range_kernel_v<ndRangeFreeFuncKernel, 2>,
                   true>("Expected true because ndRangeFreeFuncKernel() is a "
                         "single task free function kernel.");
  Failed += performCheck<syclexp::is_kernel_v<ndRangeFreeFuncKernel>, true>(
      "Expected true because ndRangeFreeFuncKernel() is a free function "
      "kernel.");

  Failed += performCheck<
      syclexp::is_nd_range_kernel_v<ns::nsNdRangeFreeFuncKernel, 3>, true>(
      "Expected true because ClassScope::staticMemberFunc() is a nd_range free "
      "function kernel.");
  Failed +=
      performCheck<syclexp::is_kernel_v<ns::nsNdRangeFreeFuncKernel>, true>(
          "Expected true because ns::nsNdRangeFreeFuncKernel() is a free "
          "function kernel.");

  Failed += performCheck<
      syclexp::is_single_task_kernel_v<ns::nsSingleTaskFreeFuncKernel>, true>(
      "Expected true because ns::nsSingleTaskFreeFuncKernel() is a single task "
      "free function kernel.");
  Failed +=
      performCheck<syclexp::is_kernel_v<ns::nsSingleTaskFreeFuncKernel>, true>(
          "Expected true because ns::nsSingleTaskFreeFuncKernel is a free "
          "function kernel.");

  return Failed;
}
