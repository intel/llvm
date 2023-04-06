// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

// TODO: Require ext_oneapi_virtual_mem aspect here when supported.

#include <sycl/sycl.hpp>

namespace syclext = sycl::ext::oneapi::experimental;

template <typename T> class VirtualVector {
public:
  VirtualVector(sycl::queue &Q)
      : MDevice{Q.get_device()}, MContext{Q.get_context()} {};

  ~VirtualVector() {
    // Free all mapped ranges.
    for (const VirtualAddressRange &VARange : MVARanges) {
      syclext::unmap(VARange.Ptr, VARange.Size, MContext);
      syclext::free_virtual_mem(VARange.Ptr, VARange.Size, MContext);
    }
    // Physical memory allocations will be freed when the physical_mem objects
    // die with MPhysicalMems.
  }

  void reserve(size_t NewSize) {
    // If we already have more memory than required, we can return.
    size_t NewByteSize = sizeof(T) * NewSize;
    if (NewByteSize <= MByteSize) {
      MSize = NewSize;
      return;
    }

    // Align the size by the minimum granularity.
    size_t AlignedNewByteSize = AlignByteSize(NewByteSize);
    size_t AlignedNewVARangeSize = AlignedNewByteSize - MByteSize;

    // Try to reserve virtual memory at the end of the existing one.
    void *CurrentEnd = reinterpret_cast<char *>(MBasePtr) + MByteSize;
    void *NewVAPtr = syclext::reserve_virtual_mem(
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
    NewPhysicalMem.map(NewVAPtr, AlignedNewVARangeSize, 0,
                       sycl::access_mode::read_write);
    MPhysicalMems.push_back(std::move(NewPhysicalMem));

    // Update the byte size of the vector.
    MSize = NewSize;
    MByteSize = AlignedNewByteSize;
  }

  size_t size() const noexcept { return MSize; }
  T *data() const noexcept { return MBasePtr; }

private:
  size_t AlignByteSize(size_t UnalignedByteSize) const {
    size_t Granularity = syclext::get_minimum_mem_granularity(
        UnalignedByteSize, MDevice, MContext);
    return ((UnalignedByteSize + Granularity - 1) / Granularity) * Granularity;
  }

  void *RecreateAddressRange(size_t AlignedNewByteSize) {
    // Reserve the full range.
    void *NewFullVAPtr =
        syclext::reserve_virtual_mem(AlignedNewByteSize, MContext);

    // Unmap the old virtual address in its entirety.
    syclext::unmap(MBasePtr, MByteSize, MContext);

    // Remap all existing ranges.
    char *NewEnd = reinterpret_cast<char *>(NewFullVAPtr);
    for (const syclext::physical_mem &PhysicalMem : MPhysicalMems) {
      PhysicalMem.map(NewEnd, PhysicalMem.size(), 0,
                      sycl::access_mode::read_write);
      NewEnd += PhysicalMem.size();
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
    VirtualAddressRange(void *Ptr, size_t Size) : Ptr{Ptr}, Size{Size} {}

    void *Ptr;
    size_t Size;
  };

  sycl::device MDevice;
  sycl::context MContext;

  std::vector<VirtualAddressRange> MVARanges;
  std::vector<syclext::physical_mem> MPhysicalMems;

  T *MBasePtr = nullptr;
  size_t MSize = 0;
  size_t MByteSize = 0;
};

constexpr size_t NumIters = 10;
constexpr size_t WriteValueOffset = 42;

int main() {
  sycl::queue Q;

  if (!Q.get_device().has(sycl::aspect::ext_oneapi_virtual_mem)) {
    std::cout
        << "sycl::aspect::ext_oneapi_virtual_mem not supported. Skipping..."
        << std::endl;
    return 0;
  }

  VirtualVector<int> Vec(Q);

  // To better test the functionality, try to allocate below the granularity
  // but enough to require more memory for some iterations.
  size_t SizeIncrement = 11;
  size_t MinSizeGran =
      syclext::get_minimum_mem_granularity(SizeIncrement * sizeof(int), Q);
  SizeIncrement = std::max(MinSizeGran / 2 - 1, SizeIncrement);

  for (size_t I = 0; I < NumIters; ++I) {
    // Increment the size of the vector.
    size_t NewVecSize = (I + 1) * SizeIncrement;
    Vec.reserve(NewVecSize);
    assert(Vec.size() == NewVecSize);

    // Populate to the new memory
    int *VecDataPtr = Vec.data();
    Q.parallel_for(sycl::range<1>{SizeIncrement}, [=](sycl::id<1> Idx) {
       VecDataPtr[I * SizeIncrement + Idx] = Idx + WriteValueOffset * (I + 1);
     }).wait_and_throw();

    // Copy back the values and verify.
    int *CopyBack = sycl::malloc_shared<int>(NewVecSize, Q);
    Q.copy(VecDataPtr, CopyBack, NewVecSize).wait_and_throw();
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
