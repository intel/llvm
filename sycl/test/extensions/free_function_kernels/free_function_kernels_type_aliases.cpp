// RUN: %clangxx -fsyntax-only -fsycl %s
// This test verifies whether type aliases aree allowed as kernel parameter to
// free function kernel.
#include <sycl/sycl.hpp>

namespace syclexp = sycl::ext::oneapi::experimental;

struct TestStruct {};
using StructAlias = TestStruct;

struct TestClass {};
using ClassAlias = TestStruct;

using WriteOnlyAcc = sycl::accessor<int, 1, sycl::access::mode::write>;

namespace ns {
using ReadOnlyAcc = sycl::accessor<float, 1, sycl::access::mode::read>;

using LocalAcc = sycl::local_accessor<int, 1>;
} // namespace ns

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(syclexp::single_task_kernel)
void singleTaskKernelStruct(StructAlias Type) {}

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(syclexp::nd_range_kernel<1>)
void ndRangeKernelStruct(StructAlias Type) {}

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(syclexp::single_task_kernel)
void singleTaskKernelClass(ClassAlias Type) {}

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(syclexp::nd_range_kernel<1>)
void ndRangeKernelClass(ClassAlias Type) {}

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(syclexp::single_task_kernel)
void singleTaskWriteOnlyAcc(WriteOnlyAcc Type) {}

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(syclexp::nd_range_kernel<1>)
void ndRangeKernelWriteOnlyAcc(WriteOnlyAcc Type) {}

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(syclexp::single_task_kernel)
void singleTaskReadOnlyAcc(ns::ReadOnlyAcc Type) {}

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(syclexp::nd_range_kernel<1>)
void ndRangeKernelReadOnlyAcc(ns::ReadOnlyAcc Type) {}

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(syclexp::single_task_kernel)
void singleTaskLocalAcc(ns::LocalAcc Type) {}

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(syclexp::nd_range_kernel<1>)
void ndRangeKernelLocalAcc(ns::LocalAcc Type) {}

template <auto Func, typename T> void runNdRangeCheck(T Type) {
  sycl::queue Queue;
  sycl::kernel_bundle Bundle =
      get_kernel_bundle<sycl::bundle_state::executable>(Queue.get_context());
  sycl::kernel_id Id = syclexp::get_kernel_id<Func>();
  sycl::kernel Kernel = Bundle.get_kernel(Id);
  Queue.submit([&](sycl::handler &Handler) {
    Handler.set_args(Type);
    Handler.parallel_for(sycl::nd_range{{1}, {1}}, Kernel);
  });
}

template <auto Func, typename T> void runSingleTaskTest(T Type) {
  sycl::queue Queue;
  sycl::kernel_bundle Bundle =
      get_kernel_bundle<sycl::bundle_state::executable>(Queue.get_context());
  sycl::kernel_id Id = syclexp::get_kernel_id<Func>();
  sycl::kernel Kernel = Bundle.get_kernel(Id);
  Queue.submit([&](sycl::handler &Handler) {
    Handler.set_args(Type);
    Handler.single_task(Kernel);
  });
}

int main() {
  runSingleTaskTest<singleTaskKernelStruct>(StructAlias());
  runSingleTaskTest<singleTaskKernelClass>(ClassAlias());
  runSingleTaskTest<singleTaskReadOnlyAcc>(ns::ReadOnlyAcc());
  runSingleTaskTest<singleTaskWriteOnlyAcc>(WriteOnlyAcc());
  runSingleTaskTest<singleTaskLocalAcc>(ns::LocalAcc());

  runNdRangeCheck<ndRangeKernelStruct>(StructAlias());
  runNdRangeCheck<ndRangeKernelStruct>(ClassAlias());
  runNdRangeCheck<ndRangeKernelReadOnlyAcc>(ns::ReadOnlyAcc());
  runNdRangeCheck<ndRangeKernelWriteOnlyAcc>(WriteOnlyAcc());
  runNdRangeCheck<ndRangeKernelLocalAcc>(ns::LocalAcc());

  return 0;
}
