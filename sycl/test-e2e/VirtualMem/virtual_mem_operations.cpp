// This test checks whether certain operations in virtual memory extension work
// as expectd.

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include "helpers.hpp"

int main() {

  constexpr size_t NumberOfIterations = 3;
  std::array<size_t, NumberOfIterations> NumberOfElementsPerIteration{10, 100,
                                                                      1000};

  sycl::queue Q;
  sycl::context Context = Q.get_context();
  sycl::device Device = Q.get_device();

  // A check should be performed that we can successfully perform and
  // immediately release a valid reservation.
  for (const size_t RequiredNumElements : NumberOfElementsPerIteration) {
    size_t BytesRequired = RequiredNumElements * sizeof(int);
    size_t UsedGranularity = GetLCMGranularity(Device, Context);
    size_t AlignedByteSize = GetAlignedByteSize(BytesRequired, UsedGranularity);
    uintptr_t VirtualMemoryPtr =
        syclext::reserve_virtual_mem(0, AlignedByteSize, Context);
    syclext::free_virtual_mem(VirtualMemoryPtr, AlignedByteSize, Context);
  }

  // A check should be performed that we can successfully map and immediately
  // unmap a virtual memory range.
  for (const size_t RequiredNumElements : NumberOfElementsPerIteration) {
    size_t BytesRequired = RequiredNumElements * sizeof(int);
    size_t UsedGranularity = GetLCMGranularity(Device, Context);
    size_t AlignedByteSize = GetAlignedByteSize(BytesRequired, UsedGranularity);
    uintptr_t VirtualMemoryPtr =
        syclext::reserve_virtual_mem(0, AlignedByteSize, Context);
    syclext::physical_mem PhysicalMem{Device, Context, AlignedByteSize};
    void *MappedPtr = PhysicalMem.map(VirtualMemoryPtr, AlignedByteSize,
                                      syclext::address_access_mode::read_write);
    syclext::unmap(MappedPtr, AlignedByteSize, Context);
    syclext::free_virtual_mem(VirtualMemoryPtr, AlignedByteSize, Context);
  }

  {
    // Check should be performed that methods get_context(), get_device() and
    // size() return correct values (i.e. ones which were passed to physical_mem
    // constructor).
    size_t BytesRequired = NumberOfElementsPerIteration[2] * sizeof(int);
    size_t UsedGranularity = GetLCMGranularity(Device, Context);
    size_t AlignedByteSize = GetAlignedByteSize(BytesRequired, UsedGranularity);

    syclext::physical_mem PhysicalMem{Device, Context, AlignedByteSize};

    assert(PhysicalMem.get_device() == Device &&
           "device passed to physical_mem must be the same as returned from "
           "get_device()");

    assert(PhysicalMem.get_context() == Context &&
           "context passed to physical_mem must be the same as returned from "
           "get_context()");

    assert(PhysicalMem.size() == AlignedByteSize &&
           "size in bytes passed to physical_mem must be the same as returned "
           "from size()");
  }

  {
    // Check to see if value returned from a valid call to map() is the same as
    // reinterpret_cast<void *>(ptr).
    size_t BytesRequired = NumberOfElementsPerIteration[2] * sizeof(int);
    size_t UsedGranularity = GetLCMGranularity(Device, Context);
    size_t AlignedByteSize = GetAlignedByteSize(BytesRequired, UsedGranularity);

    uintptr_t VirtualMemoryPtr =
        syclext::reserve_virtual_mem(0, AlignedByteSize, Context);

    syclext::physical_mem PhysicalMem{Device, Context, AlignedByteSize};

    void *MappedPtr = PhysicalMem.map(VirtualMemoryPtr, AlignedByteSize,
                                      syclext::address_access_mode::read_write);

    assert(MappedPtr == reinterpret_cast<void *>(VirtualMemoryPtr) &&
           "value returned from a valid call to map() must be equal "
           "reinterpret_cast<void *>(ptr)");

    syclext::unmap(MappedPtr, AlignedByteSize, Context);
    syclext::free_virtual_mem(VirtualMemoryPtr, AlignedByteSize, Context);
  }

  // Check to see if can change access mode of a virtual memory range and
  // immediately see it changed.
  for (const size_t RequiredNumElements : NumberOfElementsPerIteration) {
    size_t BytesRequired = RequiredNumElements * sizeof(int);
    size_t UsedGranularity = GetLCMGranularity(Device, Context);
    size_t AlignedByteSize = GetAlignedByteSize(BytesRequired, UsedGranularity);
    uintptr_t VirtualMemoryPtr =
        syclext::reserve_virtual_mem(0, AlignedByteSize, Context);
    syclext::physical_mem PhysicalMem{Device, Context, AlignedByteSize};
    void *MappedPtr = PhysicalMem.map(VirtualMemoryPtr, AlignedByteSize,
                                      syclext::address_access_mode::read_write);

    syclext::address_access_mode CurrentAccessMode =
        syclext::get_access_mode(MappedPtr, AlignedByteSize, Context);

    assert(CurrentAccessMode == syclext::address_access_mode::read_write &&
           "access mode must be address_access_mode::read_write before change "
           "with "
           "set_access_mode()");

    syclext::set_access_mode(MappedPtr, AlignedByteSize,
                             syclext::address_access_mode::read, Context);

    CurrentAccessMode =
        syclext::get_access_mode(MappedPtr, AlignedByteSize, Context);

    assert(CurrentAccessMode == syclext::address_access_mode::read &&
           "access mode must be address_access_mode::read after change with "
           "set_access_mode()");

    syclext::unmap(MappedPtr, AlignedByteSize, Context);
    syclext::free_virtual_mem(VirtualMemoryPtr, AlignedByteSize, Context);
  }

  return 0;
}
