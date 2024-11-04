// This test checks whether data can be correctly written to and read from
// virtual memory.

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include "helpers.hpp"

int main() {
  sycl::queue Q;
  sycl::context Context = Q.get_context();
  sycl::device Device = Q.get_device();
  int Failed = 0;
  constexpr size_t NumberOfElements = 1000;
  size_t BytesRequired = NumberOfElements * sizeof(int);

  size_t UsedGranularity = GetLCMGranularity(Device, Context);

  size_t AlignedByteSize =
      ((BytesRequired + UsedGranularity - 1) / UsedGranularity) *
      UsedGranularity;

  syclext::physical_mem NewPhysicalMem{Device, Context, AlignedByteSize};
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
