// REQUIRES: aspect-ext_oneapi_virtual_mem, usm_shared_allocations

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include <sycl/detail/core.hpp>
#include <sycl/usm.hpp>

#include <sycl/ext/oneapi/experimental/device_architecture.hpp>
#include <sycl/ext/oneapi/virtual_mem/physical_mem.hpp>
#include <sycl/ext/oneapi/virtual_mem/virtual_mem.hpp>

namespace syclext = sycl::ext::oneapi::experimental;

// Find the least common multiple of the context and device granularities. This
// value can be used for aligning both physical memory allocations and for
// reserving virtual memory ranges.
size_t GetLCMGranularity(const sycl::device &Dev, const sycl::context &Ctx) {
  size_t CtxGranularity = syclext::get_mem_granularity(MContext);
  size_t DevGranularity = syclext::get_mem_granularity(MDevice, MContext);

  size_t GCD = CtxGranularity;
  size_t Rem = DevGranularity % GCD;
  while (Rem != 0) {
    std::swap(GCD, Rem);
    Rem %= GCD;
  }
  return (DevGranularity / GCD) * LCMGranularity;
}

template <typename T> class VirtualVector {
public:
  VirtualVector(sycl::queue &Q)
      : MDevice{Q.get_device()}, MContext{Q.get_context()},
        MGranularity{GetLCMGranularity(MDevice, MContext)} {};

  ~VirtualVector() {
    // Free all mapped ranges.
    unmap_all();
    for (const VirtualAddressRange &VARange : MVARanges)
      syclext::free_virtual_mem(VARange.Ptr, VARange.Size, MContext);
    // Physical memory allocations will be freed when the physical_mem objects
    // die with MPhysicalMemMappings.
  }

  void reserve(size_t NewSize) {
    // If we already have more memory than required, we can return.
    size_t NewByteSize = sizeof(T) * NewSize;
    if (NewByteSize <= MByteSize) {
      MSize = NewSize;
      return;
    }

    // Align the size by the granularity.
    size_t AlignedNewByteSize = AlignByteSize(NewByteSize);
    size_t AlignedNewVARangeSize = AlignedNewByteSize - MByteSize;

    // Try to reserve virtual memory at the end of the existing one.
    uintptr_t CurrentEnd = reinterpret_cast<uintptr_t>(MBasePtr) + MByteSize;
    uintptr_t NewVAPtr = syclext::reserve_virtual_mem(
        CurrentEnd, AlignedNewVARangeSize, MContext);

    // If we failed to get a ptr to the end of the current range, we need to
    // recreate the whole range.
    if (CurrentEnd && NewVAPtr != CurrentEnd) {
      // First we need to free the virtual address range we just reserved.
      syclext::free_virtual_mem(NewVAPtr, AlignedNewVARangeSize, MContext);

      // Recreate the full range and update the new VA ptr. CurrentEnd is no
      // longer valid after this call.
      NewVAPtr = RecreateAddressRange(AlignedNewByteSize);
    } else {
      // Otherwise we need to register the new range.
      MVARanges.emplace_back(NewVAPtr, AlignedNewVARangeSize);

      // If there was no base pointer previously, this is now the new base.
      if (!MBasePtr)
        MBasePtr = reinterpret_cast<T *>(NewVAPtr);
    }

    // Create new physical memory allocation and map the new range to it.
    syclext::physical_mem NewPhysicalMem{MDevice, MContext,
                                         AlignedNewVARangeSize};
    void *MappedPtr =
        NewPhysicalMem.map(NewVAPtr, AlignedNewVARangeSize,
                           syclext::address_access_mode::read_write);
    MPhysicalMemMappings.push_back(
        std::make_pair(std::move(NewPhysicalMem), MappedPtr));

    // Update the byte size of the vector.
    MSize = NewSize;
    MByteSize = AlignedNewByteSize;
  }

  size_t size() const noexcept { return MSize; }
  T *data() const noexcept { return MBasePtr; }

private:
  size_t AlignByteSize(size_t UnalignedByteSize) const {
    return ((UnalignedByteSize + MGranularity - 1) / MGranularity) *
           MGranularity;
  }

  void unmap_all() {
    for (std::pair<syclext::physical_mem, void *> &Mapping :
         MPhysicalMemMappings) {
      if (Mapping.second == 0)
        continue;
      syclext::unmap(Mapping.second, Mapping.first.size(), MContext);
      Mapping.second = 0;
    }
  }

  uintptr_t RecreateAddressRange(size_t AlignedNewByteSize) {
    // Reserve the full range.
    uintptr_t NewFullVAPtr =
        syclext::reserve_virtual_mem(AlignedNewByteSize, MContext);

    // Unmap the old virtual address ranges.
    unmap_all();

    // Remap all existing ranges.
    uintptr_t NewEnd = NewFullVAPtr;
    for (std::pair<syclext::physical_mem, void *> &Mapping :
         MPhysicalMemMappings) {
      Mapping.second =
          Mapping.first.map(NewEnd, Mapping.first.size(),
                            syclext::address_access_mode::read_write);
      NewEnd += Mapping.first.size();
    }

    // Free the old ranges.
    for (const VirtualAddressRange &VARange : MVARanges)
      syclext::free_virtual_mem(VARange.Ptr, VARange.Size, MContext);

    // Insert the newly reserved range to the saved ranges.
    MVARanges.clear();
    MVARanges.emplace_back(NewFullVAPtr, AlignedNewByteSize);

    // Update the base pointer to point to the new start.
    MBasePtr = reinterpret_cast<T *>(NewFullVAPtr);

    // Return the new end of the mapped ranges.
    return NewEnd;
  }

  struct VirtualAddressRange {
    VirtualAddressRange(uintptr_t Ptr, size_t Size) : Ptr{Ptr}, Size{Size} {}

    uintptr_t Ptr;
    size_t Size;
  };

  sycl::device MDevice;
  sycl::context MContext;

  std::vector<VirtualAddressRange> MVARanges;
  std::vector<std::pair<syclext::physical_mem, void *>> MPhysicalMemMappings;

  T *MBasePtr = nullptr;
  size_t MSize = 0;
  size_t MByteSize = 0;

  const size_t MGranularity = 0;
};

static constexpr size_t NumIters = 10;
static constexpr size_t WriteValueOffset = 42;
static constexpr size_t NumWorkItems = 512;

int main() {
  sycl::queue Q;

  VirtualVector<int> Vec(Q);

  // To better test the functionality, try to allocate below the granularity
  // but enough to require more memory for some iterations.
  size_t SizeIncrement = 11;
  size_t MinSizeGran =
      syclext::get_mem_granularity(Q.get_device(), Q.get_context()) /
      sizeof(int);
  SizeIncrement = std::max(MinSizeGran / 2 - 1, SizeIncrement);

  // Each work-item will work on multiple elements.
  size_t NumElemsPerWI = 1 + (SizeIncrement - 1) / NumWorkItems;

  for (size_t I = 0; I < NumIters; ++I) {
    // Increment the size of the vector.
    size_t NewVecSize = (I + 1) * SizeIncrement;
    Vec.reserve(NewVecSize);
    assert(Vec.size() == NewVecSize);

    // Populate to the new memory
    int *VecDataPtr = Vec.data();
    size_t StartOffset = I * SizeIncrement;
    size_t IterWriteValueOffset = WriteValueOffset * (I + 1);
    Q.parallel_for(sycl::range<1>{NumWorkItems}, [=](sycl::item<1> Idx) {
       for (size_t J = 0; J < NumElemsPerWI; ++J) {
         size_t LoopIdx = J * Idx.get_range(0) + Idx;
         size_t OffsetIdx = StartOffset + LoopIdx;
         if (OffsetIdx < NewVecSize)
           VecDataPtr[OffsetIdx] = LoopIdx + IterWriteValueOffset;
       }
     }).wait_and_throw();

    // Copy back the values and verify.
    int *CopyBack = sycl::malloc_shared<int>(NewVecSize, Q);

    // TODO: Level-zero (excluding on PVC) does not currently allow copy across
    //       virtual memory ranges, even if they are consequtive.
    syclext::architecture DevArch =
        Q.get_device().get_info<syclext::info::device::architecture>();
    if (Q.get_backend() == sycl::backend::ext_oneapi_level_zero &&
        DevArch != syclext::architecture::intel_gpu_pvc &&
        DevArch != syclext::architecture::intel_gpu_pvc_vg) {
      Q.parallel_for(sycl::range<1>{NewVecSize}, [=](sycl::id<1> Idx) {
         CopyBack[Idx] = VecDataPtr[Idx];
       }).wait_and_throw();
    } else {
      Q.copy(VecDataPtr, CopyBack, NewVecSize).wait_and_throw();
    }

    for (size_t J = 0; J < NewVecSize; ++J) {
      int ExpectedVal =
          J % SizeIncrement + WriteValueOffset * (J / SizeIncrement + 1);
      if (CopyBack[J] != ExpectedVal) {
        std::cout << "Comparison failed at index " << J << ": " << CopyBack[J]
                  << " != " << ExpectedVal << std::endl;
        return 1;
      }
    }
    sycl::free(CopyBack, Q);
  }

  return 0;
}
