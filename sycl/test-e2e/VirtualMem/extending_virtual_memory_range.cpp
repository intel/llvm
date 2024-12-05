// This test checks whether memory accesses to contiguous virtual memory ranges
// are performed correctly

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include <cassert>

#include "helpers.hpp"

struct VirtualAddressRange {
  VirtualAddressRange(uintptr_t Ptr, size_t Size) : MPtr{Ptr}, MSize{Size} {}

  uintptr_t MPtr;
  size_t MSize;
};

struct PhysicalMemoryMapping {
  PhysicalMemoryMapping(syclext::physical_mem &&PhysicalMem, void *MappingPtr)
      : MPhysicalMem(std::move(PhysicalMem)), MMappingPtr(MappingPtr) {}
  syclext::physical_mem MPhysicalMem;
  void *MMappingPtr;
};

int main() {
  int Failed = 0;
  sycl::queue Q;
  sycl::context Context = Q.get_context();
  sycl::device Device = Q.get_device();

  constexpr size_t NumberOfVirtualMemoryRanges = 5;
  constexpr size_t ElementsInRange = 100;
  constexpr int ValueSetInKernel = 999;

  size_t BytesRequiredPerRange = ElementsInRange * sizeof(int);

  size_t UsedGranularity = GetLCMGranularity(Device, Context);

  size_t AlignedByteSizePerRange =
      GetAlignedByteSize(BytesRequiredPerRange, UsedGranularity);

  std::vector<VirtualAddressRange> VirtualMemoryRanges;
  std::vector<PhysicalMemoryMapping> PhysicalMemoryMappings;

  for (size_t Index = 0; Index < NumberOfVirtualMemoryRanges; ++Index) {
    uintptr_t VirtualMemoryPtr =
        syclext::reserve_virtual_mem(AlignedByteSizePerRange, Context);
    syclext::physical_mem PhysicalMem{Device, Context, AlignedByteSizePerRange};
    void *MappedPtr = PhysicalMem.map(VirtualMemoryPtr, AlignedByteSizePerRange,
                                      syclext::address_access_mode::read_write);

    VirtualMemoryRanges.emplace_back(VirtualMemoryPtr, AlignedByteSizePerRange);
    PhysicalMemoryMappings.emplace_back(std::move(PhysicalMem), MappedPtr);
  }

  std::vector<int> ResultHostData(ElementsInRange);

  for (size_t Index = 0; Index < NumberOfVirtualMemoryRanges; ++Index) {
    int *DataRangePtr =
        reinterpret_cast<int *>(PhysicalMemoryMappings[Index].MMappingPtr);

    Q.parallel_for(ElementsInRange, [=](sycl::id<1> Idx) {
       DataRangePtr[Idx] = ValueSetInKernel;
     }).wait_and_throw();

    {
      sycl::buffer<int> ResultBuffer(ResultHostData);

      Q.submit([&](sycl::handler &Handle) {
        sycl::accessor A(ResultBuffer, Handle, sycl::write_only);
        Handle.parallel_for(ElementsInRange, [=](sycl::id<1> Idx) {
          A[Idx] = DataRangePtr[Idx];
        });
      });
    }

    for (size_t i = 0; i < ElementsInRange; i++) {
      if (ResultHostData[i] != ValueSetInKernel) {
        std::cout << "Comparison failed with virtual range " << Index + 1
                  << " at index " << i << ": " << ResultHostData[i]
                  << " != " << ValueSetInKernel << std::endl;
        ++Failed;
      }
    }
  }

  for (auto PhysMemMap : PhysicalMemoryMappings) {
    syclext::unmap(PhysMemMap.MMappingPtr, PhysMemMap.MPhysicalMem.size(),
                   Context);
  }
  for (auto VirtualMemRange : VirtualMemoryRanges) {
    syclext::free_virtual_mem(VirtualMemRange.MPtr, VirtualMemRange.MSize,
                              Context);
  }

  return Failed;
}
