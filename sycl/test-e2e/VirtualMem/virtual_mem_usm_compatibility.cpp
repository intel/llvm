// REQUIRES: aspect-usm_shared_allocations

// This test checks whether a pointer produced by a virtual memory
// range mapping can indeed be used in various APIs accepting a USM pointer.

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include <sycl/usm.hpp>

#include <string_view>

#include "helpers.hpp"

int performResultCheck(size_t NumberOfElements, const int *DataResultPtr,
                       const int ExpectedResultValue,
                       std::string_view ErrorMessage) {
  int IsSuccessful{0};
  for (size_t i = 0; i < NumberOfElements; i++) {
    if (DataResultPtr[i] != ExpectedResultValue) {
      std::cerr << ErrorMessage << i << ": " << DataResultPtr
                << " != " << ExpectedResultValue << std::endl;
      ++IsSuccessful;
    }
  }
  return IsSuccessful;
}
int main() {

  sycl::queue Queue;
  sycl::context Context = Queue.get_context();
  sycl::device Device = Queue.get_device();

  int Failed = 0;
  constexpr int ValueSetInKernelForCopyToUSM = 111;
  constexpr int ValueSetForCopyToVirtualMem = 222;
  constexpr int ValueSetInMemSetOperationPerByte = 1;
  constexpr int ValueSetInFillOperation = 444;
  constexpr size_t NumberOfElements = 1000;

  int *CopyBack = sycl::malloc_shared<int>(NumberOfElements, Queue);
  int *CopyFrom = sycl::malloc_shared<int>(NumberOfElements, Queue);

  size_t BytesRequired = NumberOfElements * sizeof(int);

  size_t UsedGranularity = GetLCMGranularity(Device, Context);
  size_t AlignedByteSize = GetAlignedByteSize(BytesRequired, UsedGranularity);

  syclext::physical_mem PhysicalMem{Device, Context, AlignedByteSize};
  uintptr_t VirtualMemoryPtr =
      syclext::reserve_virtual_mem(0, AlignedByteSize, Context);

  void *MappedPtr = PhysicalMem.map(VirtualMemoryPtr, AlignedByteSize,
                                    syclext::address_access_mode::read_write);

  int *DataPtr = reinterpret_cast<int *>(MappedPtr);

  auto copyBackFunc = [&Queue, CopyBack, DataPtr]() {
    Queue
        .parallel_for(NumberOfElements,
                      [=](sycl::id<1> Idx) { CopyBack[Idx] = DataPtr[Idx]; })
        .wait_and_throw();
  };

  Queue
      .parallel_for(
          NumberOfElements,
          [=](sycl::id<1> Idx) { DataPtr[Idx] = ValueSetInKernelForCopyToUSM; })
      .wait_and_throw();

  // Check that one can copy from virtual memory to a USM allocation.

  copyBackFunc();
  Failed += performResultCheck(NumberOfElements, CopyBack,
                               ValueSetInKernelForCopyToUSM,
                               "Comparison failed after copy from virtual "
                               "memory to a USM allocation at index ");

  // Check that can copy from a USM allocation to virtual memory

  for (size_t Idx = 0; Idx < NumberOfElements; ++Idx) {
    CopyFrom[Idx] = ValueSetForCopyToVirtualMem;
  }

  Queue
      .parallel_for(NumberOfElements,
                    [=](sycl::id<1> Idx) { DataPtr[Idx] = CopyFrom[Idx]; })
      .wait_and_throw();

  copyBackFunc();

  Failed += performResultCheck(NumberOfElements, CopyBack,
                               ValueSetForCopyToVirtualMem,
                               "Comparison failed after copy from a USM "
                               "allocation to virtual memory at index ");

  // Check that can use memset on virtual memory
  int ExpectedResultAfterMemSetOperation{0};
  std::memset(&ExpectedResultAfterMemSetOperation,
              ValueSetInMemSetOperationPerByte, sizeof(int));
  Queue.memset(MappedPtr, ValueSetInMemSetOperationPerByte, AlignedByteSize)
      .wait_and_throw();

  copyBackFunc();

  Failed += performResultCheck(
      NumberOfElements, CopyBack, ExpectedResultAfterMemSetOperation,
      "Comparison failed after memset operation on virtual memory at index ");

  // Check that can use fill on virtual memory
  Queue.fill(DataPtr, ValueSetInFillOperation, NumberOfElements)
      .wait_and_throw();

  copyBackFunc();

  Failed += performResultCheck(
      NumberOfElements, CopyBack, ValueSetInFillOperation,
      "Comparison failed after fill operation on virtual memory at index ");

  sycl::free(CopyFrom, Queue);
  sycl::free(CopyBack, Queue);
  syclext::unmap(MappedPtr, AlignedByteSize, Context);
  syclext::free_virtual_mem(VirtualMemoryPtr, AlignedByteSize, Context);

  return Failed;
}
