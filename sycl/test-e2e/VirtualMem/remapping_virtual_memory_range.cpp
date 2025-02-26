// This test checks whether virtual memory range can correctly be accessed
// even if it was re-mapped to a different physical range.

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include <cassert>

#include "helpers.hpp"

namespace syclext = sycl::ext::oneapi::experimental;

int main() {

  sycl::queue Q;
  sycl::context Context = Q.get_context();
  sycl::device Device = Q.get_device();

  int Failed = 0;

  constexpr size_t NumberOfElements = 1000;
  constexpr int ValueSetInFirstKernel = 555;
  constexpr int ValueSetInSecondKernel = 999;

  size_t BytesRequired = NumberOfElements * sizeof(int);

  size_t UsedGranularity = GetLCMGranularity(Device, Context);
  size_t AlignedByteSize = GetAlignedByteSize(BytesRequired, UsedGranularity);

  syclext::physical_mem FirstPhysicalMemory{Device, Context, AlignedByteSize};
  uintptr_t VirtualMemoryPtr =
      syclext::reserve_virtual_mem(0, AlignedByteSize, Context);

  void *MappedPtr =
      FirstPhysicalMemory.map(VirtualMemoryPtr, AlignedByteSize,
                              syclext::address_access_mode::read_write);

  int *DataPtr = reinterpret_cast<int *>(MappedPtr);

  std::vector<int> ResultHostData(NumberOfElements);

  Q.parallel_for(NumberOfElements, [=](sycl::id<1> Idx) {
     DataPtr[Idx] = ValueSetInFirstKernel;
   }).wait_and_throw();

  syclext::unmap(MappedPtr, AlignedByteSize, Context);

  syclext::physical_mem SecondPhysicalMemory{Device, Context, AlignedByteSize};
  MappedPtr =
      SecondPhysicalMemory.map(VirtualMemoryPtr, AlignedByteSize,
                               syclext::address_access_mode::read_write);

  Q.parallel_for(NumberOfElements, [=](sycl::id<1> Idx) {
     DataPtr[Idx] = ValueSetInSecondKernel;
   }).wait_and_throw();

  {
    sycl::buffer<int> ResultBuffer(ResultHostData);

    Q.submit([&](sycl::handler &Handle) {
      sycl::accessor A(ResultBuffer, Handle, sycl::write_only);
      Handle.parallel_for(NumberOfElements,
                          [=](sycl::id<1> Idx) { A[Idx] = DataPtr[Idx]; });
    });
  }

  for (size_t i = 0; i < NumberOfElements; i++) {
    if (ResultHostData[i] != ValueSetInSecondKernel) {
      std::cout << "Comparison failed at index " << i << ": "
                << ResultHostData[i] << " != " << ValueSetInSecondKernel
                << std::endl;
      ++Failed;
    }
  }

  syclext::unmap(MappedPtr, AlignedByteSize, Context);
  syclext::free_virtual_mem(VirtualMemoryPtr, AlignedByteSize, Context);

  return Failed;
}
