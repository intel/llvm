// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// This test verifies whether sycl::local_accessor can be used with free
// function kernels extension.

#include <sycl/atomic_ref.hpp>
#include <sycl/ext/oneapi/free_function_queries.hpp>
#include <sycl/group_barrier.hpp>

#include "helpers.hpp"

constexpr size_t BIN_SIZE = 4;
constexpr size_t NUM_BINS = 4;
constexpr size_t INPUT_SIZE = 1024;

namespace ns {
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void nsNdRangeFreeFunc(sycl::accessor<int, 1> InputAccessor,
                       sycl::accessor<int, 1> ResultAccessor,
                       sycl::local_accessor<int, 1> LocalAccessor) {

  size_t LocalWorkItemId =
      syclext::this_work_item::get_nd_item<1>().get_local_id();
  size_t GlobalWorkItemId =
      syclext::this_work_item::get_nd_item<1>().get_global_id();
  sycl::group<1> WorkGroup = syclext::this_work_item::get_work_group<1>();

  if (LocalWorkItemId < BIN_SIZE)
    LocalAccessor[LocalWorkItemId] = 0;

  sycl::group_barrier(WorkGroup);

  int Value = InputAccessor[GlobalWorkItemId];
  sycl::atomic_ref<int, sycl::memory_order::relaxed,
                   sycl::memory_scope::work_group>
      AtomicRefLocal(LocalAccessor[Value]);
  AtomicRefLocal++;
  sycl::group_barrier(WorkGroup);

  if (LocalWorkItemId < BIN_SIZE) {
    sycl::atomic_ref<int, sycl::memory_order::relaxed,
                     sycl::memory_scope::device>
        AtomicRefGlobal(ResultAccessor[LocalWorkItemId]);
    AtomicRefGlobal.fetch_add(LocalAccessor[LocalWorkItemId]);
  }
}
} // namespace ns

// TODO: Need to add checks for a static member functions of a class as free
// function kerenl

void FillWithData(std::vector<int> &Data, std::vector<int> &Values) {
  constexpr size_t Offset = INPUT_SIZE / NUM_BINS;
  for (size_t i = 0; i < NUM_BINS; ++i) {
    std::fill(Data.begin() + i * Offset, Data.begin() + (i + 1) * Offset,
              Values[i]);
  }
}

int main() {

  int Failed = 0;
  sycl::queue Queue;
  sycl::context Context = Queue.get_context();
  {
    // Check that sycl::local_accesor is supported inside nd_range free function
    // kernel.
    std::vector<int> ExpectedHistogramNumbers = {0, 1, 2, 3};
    std::vector<int> ResultData(BIN_SIZE, 0);

    std::vector<int> InputData(INPUT_SIZE);
    FillWithData(InputData, ExpectedHistogramNumbers);
    {
      sycl::buffer<int, 1> InputBuffer(InputData);
      sycl::buffer<int, 1> ResultBuffer(ResultData);
      sycl::kernel UsedKernel = getKernel<ns::nsNdRangeFreeFunc>(Context);
      Queue.submit([&](sycl::handler &Handler) {
        sycl::accessor<int, 1> InputAccessor{InputBuffer, Handler};
        sycl::accessor<int, 1> ResultsAccessor{ResultBuffer, Handler};
        sycl::local_accessor<int> LocalMemPerWG(sycl::range<1>(BIN_SIZE),
                                                Handler);
        Handler.set_args(InputAccessor, ResultsAccessor, LocalMemPerWG);
        sycl::nd_range<1> Ndr{INPUT_SIZE, INPUT_SIZE / NUM_BINS};
        Handler.parallel_for(Ndr, UsedKernel);
      });
    }
    Failed +=
        performResultCheck(NUM_BINS, ResultData.data(),
                           "sycl::nd_range_kernel with sycl::local_accessor",
                           INPUT_SIZE / NUM_BINS);
  }
  return Failed;
}
