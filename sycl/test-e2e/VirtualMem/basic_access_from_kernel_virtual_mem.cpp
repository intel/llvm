// REQUIRES: aspect-ext_oneapi_virtual_mem

// This test checks whether data can be correctly written to and read from
// virtual memory.

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/virtual_mem/physical_mem.hpp>
#include <sycl/ext/oneapi/virtual_mem/virtual_mem.hpp>

namespace syclext = sycl::ext::oneapi::experimental;

int main() {
  sycl::queue Q;
  sycl::context Context = Q.get_context();
  int Failed = 0;
  constexpr size_t NumberOfElements = 1000;
  size_t BytesRequired = NumberOfElements * sizeof(int);

  size_t CtxGranularity = syclext::get_mem_granularity(
      Context, syclext::granularity_mode::recommended);

  size_t AlignedByteSize =
      ((BytesRequired + CtxGranularity - 1) / CtxGranularity) * CtxGranularity;

  syclext::physical_mem NewPhysicalMem{Q.get_device(), Context,
                                       AlignedByteSize};
  uintptr_t VirtualMemoryPtr =
      syclext::reserve_virtual_mem(0, AlignedByteSize, Context);

  void *MappedPtr =
      NewPhysicalMem.map(VirtualMemoryPtr, AlignedByteSize,
                         syclext::address_access_mode::read_write);

  int *DataPtr = reinterpret_cast<int *>(MappedPtr);

  std::vector<int> ResultHostData(NumberOfElements);

  constexpr int ExpectedValueAfterFill = 1;

  Q.fill(DataPtr, ExpectedValueAfterFill, NumberOfElements).wait_and_throw();
  {
    sycl::buffer<int> CheckBuffer(ResultHostData);
    Q.submit([&](sycl::handler &Handle) {
      sycl::accessor A(CheckBuffer, Handle, sycl::write_only);
      Handle.parallel_for(NumberOfElements,
                          [=](sycl::id<1> Idx) { A[Idx] = DataPtr[Idx]; });
    });
  }

  for (size_t i = 0; i < ResultHostData.size(); i++) {
    if (ResultHostData[i] != ExpectedValueAfterFill) {
      std::cout << "Comparison failed after fill operation at index " << i
                << ": " << ResultHostData[i] << " != " << ExpectedValueAfterFill
                << std::endl;
      ++Failed;
    }
  }

  Q.parallel_for(NumberOfElements, [=](sycl::id<1> Idx) {
     DataPtr[Idx] = Idx;
   }).wait_and_throw();

  syclext::set_access_mode(DataPtr, AlignedByteSize,
                           syclext::address_access_mode::read, Context);

  {
    sycl::buffer<int> ResultBuffer(ResultHostData);

    Q.submit([&](sycl::handler &Handle) {
      sycl::accessor A(ResultBuffer, Handle, sycl::write_only);
      Handle.parallel_for(NumberOfElements,
                          [=](sycl::id<1> Idx) { A[Idx] = DataPtr[Idx]; });
    });
  }

  for (size_t i = 0; i < NumberOfElements; i++) {
    const int ExpectedValue = static_cast<int>(i);
    if (ResultHostData[i] != ExpectedValue) {
      std::cout << "Comparison failed at index " << i << ": "
                << ResultHostData[i] << " != " << ExpectedValue << std::endl;
      ++Failed;
    }
  }

  syclext::unmap(MappedPtr, AlignedByteSize, Context);
  syclext::free_virtual_mem(VirtualMemoryPtr, AlignedByteSize, Context);

  return Failed;
}
