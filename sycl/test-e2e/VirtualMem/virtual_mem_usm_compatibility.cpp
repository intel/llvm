// REQUIRES: aspect-usm_shared_allocations

// This test checks whether a pointer produced by a virtual memory
// range mapping can indeed be used in various APIs accepting a USM pointer

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include <sycl/usm.hpp>

#include "helpers.hpp"

int main() {

  sycl::queue Queue;
  sycl::context Context = Queue.get_context();
  sycl::device Device = Queue.get_device();

  int Failed = 0;
  constexpr int ValueSetInKernelForCopyToUSM = 111;
  constexpr int ValueSetForCopyToVirtualMem = 222;
  constexpr int ValueSetInMemSetOperation = 333;
  constexpr int ValueSetInFillOperation = 444;
  constexpr size_t NumberOfElements = 1000;
  size_t BytesRequired = NumberOfElements * sizeof(int);

  size_t UsedGranularity = GetLCMGranularity(Device, Context);
  size_t AlignedByteSize = GetAlignedByteSize(BytesRequired, UsedGranularity);

  syclext::physical_mem PhysicalMem{Device, Context, AlignedByteSize};
  uintptr_t VirtualMemoryPtr =
      syclext::reserve_virtual_mem(0, AlignedByteSize, Context);

  void *MappedPtr = PhysicalMem.map(VirtualMemoryPtr, AlignedByteSize,
                                    syclext::address_access_mode::read_write);

  int *DataPtr = reinterpret_cast<int *>(MappedPtr);

  Queue
      .parallel_for(
          NumberOfElements,
          [=](sycl::id<1> Idx) { DataPtr[Idx] = ValueSetInKernelForCopyToUSM; })
      .wait_and_throw();

  // Check that can copy from virtual memory to a USM allocation
  int *CopyBack = sycl::malloc_shared<int>(NumberOfElements, Queue);

  Queue
      .parallel_for(NumberOfElements,
                    [=](sycl::id<1> Idx) { CopyBack[Idx] = DataPtr[Idx]; })
      .wait_and_throw();

  for (size_t i = 0; i < NumberOfElements; i++) {
    if (CopyBack[i] != ValueSetInKernelForCopyToUSM) {
      std::cout << "Comparison failed after copy from virtual memory to a USM "
                   "allocation at index "
                << i << ": " << CopyBack[i]
                << " != " << ValueSetInKernelForCopyToUSM << std::endl;
      ++Failed;
    }
  }

  // Check that can copy from a USM allocation to virtual memory
  int *CopyFrom = sycl::malloc_shared<int>(NumberOfElements, Queue);
  for (size_t Idx = 0; Idx < NumberOfElements; ++Idx) {
    CopyFrom[Idx] = ValueSetForCopyToVirtualMem;
  }

  Queue
      .parallel_for(NumberOfElements,
                    [=](sycl::id<1> Idx) { DataPtr[Idx] = CopyFrom[Idx]; })
      .wait_and_throw();

  Queue
      .parallel_for(NumberOfElements,
                    [=](sycl::id<1> Idx) { CopyBack[Idx] = DataPtr[Idx]; })
      .wait_and_throw();

  for (size_t i = 0; i < NumberOfElements; i++) {
    if (CopyBack[i] != ValueSetForCopyToVirtualMem) {
      std::cout << "Comparison failed after copy from a USM allocation to "
                   "virtual memory at index "
                << i << ": " << CopyBack[i]
                << " != " << ValueSetForCopyToVirtualMem << std::endl;
      ++Failed;
    }
  }

  // Check that can use memset on virtual memory
  Queue.memset(DataPtr, ValueSetInMemSetOperation, AlignedByteSize)
      .wait_and_throw();

  Queue
      .parallel_for(NumberOfElements,
                    [=](sycl::id<1> Idx) { CopyBack[Idx] = DataPtr[Idx]; })
      .wait_and_throw();

  for (size_t i = 0; i < NumberOfElements; i++) {
    if (CopyBack[i] != ValueSetInMemSetOperation) {
      std::cout << "Comparison failed after memset operation on virtual memory "
                   "at index "
                << i << ": " << CopyBack[i]
                << " != " << ValueSetInMemSetOperation << std::endl;
      ++Failed;
    }
  }

  // Check that can use fill on virtual memory

  Queue.fill(DataPtr, ValueSetInFillOperation, AlignedByteSize)
      .wait_and_throw();

  Queue
      .parallel_for(NumberOfElements,
                    [=](sycl::id<1> Idx) { CopyBack[Idx] = DataPtr[Idx]; })
      .wait_and_throw();

  for (size_t i = 0; i < NumberOfElements; i++) {
    if (CopyBack[i] != ValueSetInFillOperation) {
      std::cout << "Comparison failed after fill operation on virtual memory "
                   "at index "
                << i << ": " << CopyBack[i] << " != " << ValueSetInFillOperation
                << std::endl;
      ++Failed;
    }
  }

  sycl::free(CopyFrom, Queue);
  sycl::free(CopyBack, Queue);
  syclext::unmap(MappedPtr, AlignedByteSize, Context);
  syclext::free_virtual_mem(VirtualMemoryPtr, AlignedByteSize, Context);

  return Failed;
}