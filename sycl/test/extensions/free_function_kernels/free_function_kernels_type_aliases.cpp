// RUN: %clangxx -fsyntax-only -fsycl %s
// This test verifies whether type aliases are allowed as kernel parameter to
// free function kernel.
#include <sycl/sycl.hpp>

namespace syclexp = sycl::ext::oneapi::experimental;

struct TestStruct {};
using StructAlias = TestStruct;

class TestClass {};
using ClassAlias = TestClass;
namespace ns1 {
using WriteOnlyAcc = sycl::accessor<int, 1, sycl::access::mode::write>;

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(syclexp::single_task_kernel)
void singleTaskKernelStruct(StructAlias Type) {}

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(syclexp::nd_range_kernel<1>)
void ndRangeKernelStruct(StructAlias Type) {}

namespace ns2 {

template <typename A, typename B, typename C, typename D> struct TestStructY {};

template <typename A, typename B>
using AliasTypeY = TestStructY<A, B, int, float>;

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(syclexp::single_task_kernel)
void singleTaskKernelClass(ClassAlias Type) {}

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(syclexp::nd_range_kernel<1>)
void ndRangeKernelClass(ClassAlias Type) {}

namespace ns3 {

using LocalAcc = sycl::local_accessor<int, 1>;
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(syclexp::single_task_kernel)
void singleTaskWriteOnlyAcc(WriteOnlyAcc Type) {}

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(syclexp::nd_range_kernel<1>)
void ndRangeKernelWriteOnlyAcc(WriteOnlyAcc Type) {}

namespace ns4 {
template <typename A, typename B, typename C> struct TestStructX {};
using AliasTypeX = TestStructX<int, float, int>;
} // namespace ns4
} // namespace ns3
} // namespace ns2
} // namespace ns1

namespace ns5 {
using ReadOnlyAcc = sycl::accessor<float, 1, sycl::access::mode::read>;

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(syclexp::single_task_kernel)
void singleTaskReadOnlyAcc(ReadOnlyAcc Type) {}

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(syclexp::nd_range_kernel<1>)
void ndRangeKernelReadOnlyAcc(ReadOnlyAcc Type) {}

} // namespace ns5

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(syclexp::single_task_kernel)
void singleTaskLocalAcc(ns1::ns2::ns3::LocalAcc Type) {}

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(syclexp::nd_range_kernel<1>)
void ndRangeKernelLocalAcc(ns1::ns2::ns3::LocalAcc Type) {}

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(syclexp::single_task_kernel)
void singleTaskTemplatedType(ns1::ns2::ns3::ns4::AliasTypeX Type) {}

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(syclexp::nd_range_kernel<1>)
void ndRangeKernelTemplatedType(ns1::ns2::ns3::ns4::AliasTypeX Type) {}

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(syclexp::single_task_kernel)
void singleTaskTemplatedTypeAliasTemplatedType(
    ns1::ns2::AliasTypeY<int, int> Type) {}

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(syclexp::nd_range_kernel<1>)
void ndRangeKernelTemplatedTypeAliasTemplatedType(
    ns1::ns2::AliasTypeY<float, float> Type) {}

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
  runSingleTaskTest<ns1::singleTaskKernelStruct>(StructAlias());
  runSingleTaskTest<ns1::ns2::singleTaskKernelClass>(ClassAlias());
  runSingleTaskTest<ns5::singleTaskReadOnlyAcc>(ns5::ReadOnlyAcc());
  runSingleTaskTest<ns1::ns2::ns3::singleTaskWriteOnlyAcc>(ns1::WriteOnlyAcc());
  runSingleTaskTest<singleTaskLocalAcc>(ns1::ns2::ns3::LocalAcc());
  runSingleTaskTest<singleTaskTemplatedType>(ns1::ns2::ns3::ns4::AliasTypeX());
  runSingleTaskTest<singleTaskTemplatedTypeAliasTemplatedType>(
      ns1::ns2::AliasTypeY<int, int>());

  runNdRangeCheck<ns1::ndRangeKernelStruct>(StructAlias());
  runNdRangeCheck<ns1::ns2::singleTaskKernelClass>(ClassAlias());
  runNdRangeCheck<ns5::ndRangeKernelReadOnlyAcc>(ns5::ReadOnlyAcc());
  runNdRangeCheck<ns1::ns2::ns3::ndRangeKernelWriteOnlyAcc>(
      ns1::WriteOnlyAcc());
  runNdRangeCheck<ndRangeKernelLocalAcc>(ns1::ns2::ns3::LocalAcc());
  runNdRangeCheck<ndRangeKernelTemplatedType>(ns1::ns2::ns3::ns4::AliasTypeX());
  runNdRangeCheck<ndRangeKernelTemplatedTypeAliasTemplatedType>(
      ns1::ns2::AliasTypeY<float, float>());

  return 0;
}
