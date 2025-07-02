// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// This test verifies whether struct that contains either sycl::local_accesor or
// sycl::accessor can be used with free function kernels extension.

// XFAIL: *
// XFAIL-TRACKER: CMPLRLLVM-67737

#include <sycl/atomic_ref.hpp>
#include <sycl/ext/oneapi/free_function_queries.hpp>
#include <sycl/group_barrier.hpp>

#include "helpers.hpp"

namespace ns {
// TODO: Need to remove explicit specified default template arguments for the
// accessor when the relevant CMPLRLLVM-68249 issue is fixed.
template <size_t Dims> struct StructWithAccessor {
  sycl::accessor<int, Dims, sycl::access::mode::read_write,
                 sycl::access::target::device,
                 sycl::access::placeholder::false_t>
      MAccessor;
  int MValue;
};

template <size_t Dims> struct NestedStructWithAccessor {
  StructWithAccessor<Dims> NestedStruct;
};

template <int Dims>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<Dims>))
void nsNdRangeFreeFunc(StructWithAccessor<Dims> Type) {
  auto Item = syclext::this_work_item::get_nd_item<Dims>().get_global_id();
  Type.MAccessor[Item] = Type.MValue;
}

template <int Dims>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<Dims>))
void nsNdRangeFreeFuncWithNestedStruct(NestedStructWithAccessor<Dims> Type) {
  auto Item = syclext::this_work_item::get_nd_item<Dims>().get_global_id();
  Type.NestedStruct.MAccessor[Item] = Type.NestedStruct.MValue;
}
} // namespace ns

// TODO: Need to remove explicit specified default template arguments for the
// accessor when the relevant CMPLRLLVM-68249 issue is fixed.
template <size_t Dims> struct StructWithMultipleAccessors {
  sycl::accessor<int, Dims, sycl::access::mode::read,
                 sycl::access::target::device,
                 sycl::access::placeholder::false_t>
      MInputAAcc;
  sycl::accessor<int, Dims, sycl::access::mode::read,
                 sycl::access::target::device,
                 sycl::access::placeholder::false_t>
      MInputBAcc;
  sycl::accessor<int, Dims, sycl::access::mode::write,
                 sycl::access::target::device,
                 sycl::access::placeholder::false_t>
      MResultAcc;
};

template <int Dims>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::single_task_kernel))
void globalScopeSingleFreeFunc(ns::StructWithAccessor<Dims> Type) {
  for (auto &Elem : Type.MAccessor)
    Elem = Type.MValue;
}

template <int Dims>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<Dims>))
void ndRangeFreeFuncMultipleParameters(StructWithMultipleAccessors<Dims> Type) {
  auto Item = syclext::this_work_item::get_nd_item<Dims>().get_global_id();
  Type.MResultAcc[Item] = Type.MInputAAcc[Item] + Type.MInputBAcc[Item];
}

template <auto Func, size_t Dims, bool IsNestedStruct = false>
int runNdRangeTest(sycl::queue &Queue, sycl::context &Context,
                   sycl::nd_range<Dims> NdRange, std::string_view ErrorMessage,
                   const int ExpectedResultValue) {
  sycl::kernel UsedKernel = getKernel<Func>(Context);
  std::vector<int> ResultData(NdRange.get_global_range().size(), 0);
  {
    sycl::buffer<int, Dims> Buffer(ResultData.data(),
                                   NdRange.get_global_range());
    Queue.submit([&](sycl::handler &Handler) {
      if constexpr (IsNestedStruct) {
        Handler.set_args(
            ns::NestedStructWithAccessor<Dims>{ns::StructWithAccessor<Dims>{
                sycl::accessor<int, Dims>{Buffer, Handler},
                ExpectedResultValue}});
      } else {
        Handler.set_args(ns::StructWithAccessor<Dims>{
            sycl::accessor<int, Dims>{Buffer, Handler}, ExpectedResultValue});
      }
      Handler.parallel_for(NdRange, UsedKernel);
    });
  }
  return performResultCheck(NdRange.get_global_range().size(),
                            ResultData.data(), ErrorMessage,
                            ExpectedResultValue);
}

template <auto Func, size_t Dims>
int runSingleTaskTest(sycl::queue &Queue, sycl::context &Context,
                      sycl::range<Dims> NumOfElementsPerDim,
                      std::string_view ErrorMessage,
                      const int ExpectedResultValue) {
  sycl::kernel UsedKernel = getKernel<Func>(Context);
  std::vector<int> ResultData(NumOfElementsPerDim.size(), 0);
  {
    sycl::buffer<int, Dims> Buffer(ResultData.data(), NumOfElementsPerDim);
    Queue.submit([&](sycl::handler &Handler) {
      Handler.set_arg(0, ns::StructWithAccessor<Dims>{
                             sycl::accessor<int, Dims>{Buffer, Handler},
                             ExpectedResultValue});
      Handler.single_task(UsedKernel);
    });
  }
  return performResultCheck(NumOfElementsPerDim.size(), ResultData.data(),
                            ErrorMessage, ExpectedResultValue);
}

template <auto Func, size_t Dims>
int runNdRangeTestMultipleParameters(sycl::queue &Queue, sycl::context &Context,
                                     sycl::nd_range<Dims> NdRange,
                                     std::string_view ErrorMessage,
                                     sycl::range<3> Values) {
  sycl::kernel UsedKernel = getKernel<Func>(Context);
  std::vector<int> InputAData(NdRange.get_global_range().size(), Values[0]);
  std::vector<int> InputBData(NdRange.get_global_range().size(), Values[1]);
  std::vector<int> ResultData(NdRange.get_global_range().size(), 0);

  {
    sycl::buffer<int, Dims> InputABuffer(InputAData.data(),
                                         NdRange.get_global_range());
    sycl::buffer<int, Dims> InputBBuffer(InputBData.data(),
                                         NdRange.get_global_range());
    sycl::buffer<int, Dims> ResultBuffer(ResultData.data(),
                                         NdRange.get_global_range());
    Queue.submit([&](sycl::handler &Handler) {
      Handler.set_args(StructWithMultipleAccessors<Dims>{
          sycl::accessor<int, Dims, sycl::access::mode::read,
                         sycl::access::target::device>{InputABuffer, Handler},
          sycl::accessor<int, Dims, sycl::access::mode::read,
                         sycl::access::target::device>{InputBBuffer, Handler},
          sycl::accessor<int, Dims, sycl::access::mode::write>{ResultBuffer,
                                                               Handler}});
      Handler.parallel_for(NdRange, UsedKernel);
    });
  }
  return performResultCheck(NdRange.get_global_range().size(),
                            ResultData.data(), ErrorMessage, Values[2]);
}

namespace local_acc {

constexpr size_t BIN_SIZE = 4;
constexpr size_t NUM_BINS = 4;
constexpr size_t INPUT_SIZE = 1024;

struct StructWithLocalAccessor {
  // TODO: Need to remove explicit specified default template arguments for the
  // accessor when the relevant CMPLRLLVM-68249 issue is fixed.
  sycl::accessor<int, 1, sycl::access::mode::read_write,
                 sycl::access::target::device,
                 sycl::access::placeholder::false_t>
      MInputAccessor;
  sycl::accessor<int, 1, sycl::access::mode::read_write,
                 sycl::access::target::device,
                 sycl::access::placeholder::false_t>
      MResultAccessor;
  sycl::local_accessor<int, 1> MLocalAccessor;
};

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void nsNdRangeFreeFunc(StructWithLocalAccessor Type) {

  size_t LocalWorkItemId =
      syclext::this_work_item::get_nd_item<1>().get_local_id();
  size_t GlobalWorkItemId =
      syclext::this_work_item::get_nd_item<1>().get_global_id();
  sycl::group<1> WorkGroup = syclext::this_work_item::get_work_group<1>();

  if (LocalWorkItemId < BIN_SIZE)
    Type.MLocalAccessor[LocalWorkItemId] = 0;

  sycl::group_barrier(WorkGroup);

  int Value = Type.MInputAccessor[GlobalWorkItemId];
  sycl::atomic_ref<int, sycl::memory_order::relaxed,
                   sycl::memory_scope::work_group>
      AtomicRefLocal(Type.MLocalAccessor[Value]);
  AtomicRefLocal++;
  sycl::group_barrier(WorkGroup);

  if (LocalWorkItemId < BIN_SIZE) {
    sycl::atomic_ref<int, sycl::memory_order::relaxed,
                     sycl::memory_scope::device>
        AtomicRefGlobal(Type.MResultAccessor[LocalWorkItemId]);
    AtomicRefGlobal.fetch_add(Type.MLocalAccessor[LocalWorkItemId]);
  }
}

void FillWithData(std::vector<int> &Data, std::vector<int> &Values) {
  constexpr size_t Offset = INPUT_SIZE / NUM_BINS;
  for (size_t i = 0; i < NUM_BINS; ++i) {
    std::fill(Data.begin() + i * Offset, Data.begin() + (i + 1) * Offset,
              Values[i]);
  }
}

} // namespace local_acc

int main() {

  int Failed = 0;
  sycl::queue Queue;
  sycl::context Context = Queue.get_context();

  {
    // Check struct type that contains sycl::accessor is supported inside
    // single_task free function kernel
    Failed += runSingleTaskTest<globalScopeSingleFreeFunc<1>, 1>(
        Queue, Context, sycl::range<1>{10},
        "globalScopeSingleFreeFunc with sycl::accessor<1>", 1);
    Failed += runSingleTaskTest<globalScopeSingleFreeFunc<2>, 2>(
        Queue, Context, sycl::range<2>{10, 10},
        "globalScopeSingleFreeFunc with sycl::accessor<2>", 2);
    Failed += runSingleTaskTest<globalScopeSingleFreeFunc<3>, 3>(
        Queue, Context, sycl::range<3>{5, 5, 5},
        "globalScopeSingleFreeFunc with sycl::accessor<3>", 3);
  }

  {
    // Check struct type that contains sycl::accessor is supported inside
    // nd_range free function kernel
    Failed += runNdRangeTest<ns::nsNdRangeFreeFunc<1>, 1>(
        Queue, Context, sycl::nd_range{sycl::range{10}, sycl::range{2}},
        "ns::nsNdRangeFreeFunc with struct that contains sycl::accessor<1>", 4);
    Failed += runNdRangeTest<ns::nsNdRangeFreeFunc<2>, 2>(
        Queue, Context, sycl::nd_range{sycl::range{16, 16}, sycl::range{4, 4}},
        "ns::nsNdRangeFreeFunc with struct that contains sycl::accessor<2>", 5);
    Failed += runNdRangeTest<ns::nsNdRangeFreeFunc<3>, 3>(
        Queue, Context,
        sycl::nd_range{sycl::range{10, 10, 10}, sycl::range{2, 2, 2}},
        "ns::nsNdRangeFreeFunc with struct that contains sycl::accessor<3>", 6);
  }

  {
    // Check struct type that contains multiple sycl::accessor is supported
    // inside nd_range free function kernel
    Failed +=
        runNdRangeTestMultipleParameters<ndRangeFreeFuncMultipleParameters<1>,
                                         1>(
            Queue, Context, sycl::nd_range{sycl::range{10}, sycl::range{2}},
            "ndRangeFreeFuncMultipleParameters with struct type that contains "
            "multiple sycl::accessor<1>",
            sycl::range{111, 111, 222});
    Failed +=
        runNdRangeTestMultipleParameters<ndRangeFreeFuncMultipleParameters<2>,
                                         2>(
            Queue, Context,
            sycl::nd_range{sycl::range{16, 16}, sycl::range{4, 4}},
            "ndRangeFreeFuncMultipleParameters with struct type that contains "
            "multiple sycl::accessor<2>",
            sycl::range{222, 222, 444});
    Failed +=
        runNdRangeTestMultipleParameters<ndRangeFreeFuncMultipleParameters<3>,
                                         3>(
            Queue, Context,
            sycl::nd_range{sycl::range{10, 10, 10}, sycl::range{2, 2, 2}},
            "ndRangeFreeFuncMultipleParameters with struct type that contains "
            "multiple sycl::accessor<3>",
            sycl::range{444, 444, 888});
  }

  {
    // Check struct type that nests another struct which contains sycl::accessor
    // is supported inside nd_range free function kernel
    Failed += runNdRangeTest<ns::nsNdRangeFreeFuncWithNestedStruct<1>, 1, true>(
        Queue, Context, sycl::nd_range{sycl::range{10}, sycl::range{2}},
        "ns::nsNdRangeFreeFuncWithNestedStruct with a struct nesting another "
        "struct that contains sycl::accessor<1>",
        7);
    Failed += runNdRangeTest<ns::nsNdRangeFreeFuncWithNestedStruct<2>, 2, true>(
        Queue, Context, sycl::nd_range{sycl::range{16, 16}, sycl::range{4, 4}},
        "ns::nsNdRangeFreeFuncWithNestedStruct with a struct nesting another "
        "struct that contains sycl::accessor<2>",
        8);
    Failed += runNdRangeTest<ns::nsNdRangeFreeFuncWithNestedStruct<3>, 3, true>(
        Queue, Context,
        sycl::nd_range{sycl::range{10, 10, 10}, sycl::range{2, 2, 2}},
        "ns::nsNdRangeFreeFuncWithNestedStruct with a struct nesting another "
        "struct that contains sycl::accessor<3>",
        9);
  }

  {
    // Check struct type that contains sycl::local_accesor is supported inside
    // nd_range free function kernel.
    std::vector<int> ExpectedHistogramNumbers = {0, 1, 2, 3};
    std::vector<int> ResultData(local_acc::BIN_SIZE, 0);

    std::vector<int> InputData(local_acc::INPUT_SIZE);
    local_acc::FillWithData(InputData, ExpectedHistogramNumbers);
    {
      sycl::buffer<int, 1> InputBuffer(InputData);
      sycl::buffer<int, 1> ResultBuffer(ResultData);
      sycl::kernel UsedKernel =
          getKernel<local_acc::nsNdRangeFreeFunc>(Context);
      Queue.submit([&](sycl::handler &Handler) {
        Handler.set_args(local_acc::StructWithLocalAccessor{
            sycl::accessor<int, 1>{InputBuffer, Handler},
            sycl::accessor<int, 1>{ResultBuffer, Handler},
            sycl::local_accessor<int>{sycl::range<1>(local_acc::BIN_SIZE),
                                      Handler}});
        sycl::nd_range<1> Ndr{local_acc::INPUT_SIZE,
                              local_acc::INPUT_SIZE / local_acc::NUM_BINS};
        Handler.parallel_for(Ndr, UsedKernel);
      });
    }
    Failed += performResultCheck(local_acc::NUM_BINS, ResultData.data(),
                                 "sycl::nd_range_kernel with struct type that "
                                 "contains sycl::local_accesor",
                                 local_acc::INPUT_SIZE / local_acc::NUM_BINS);
  }

  return Failed;
}
