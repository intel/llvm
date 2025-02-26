// REQUIRES: aspect-ext_oneapi_virtual_mem, linux, (gpu && level_zero)
// RUN: %{build} -o %t.out
// RUN: %{run} NEOReadDebugKeys=1 CreateMultipleRootDevices=2 %t.out

// Test for the assumption behide DevASAN shadow memory for L0GPU , which is it
// is okay to access VirtualMem from different device/context.

#include <sycl/detail/core.hpp>

#include <sycl/ext/oneapi/virtual_mem/physical_mem.hpp>
#include <sycl/ext/oneapi/virtual_mem/virtual_mem.hpp>

namespace syclext = sycl::ext::oneapi::experimental;

// Find the least common multiple of the context and device granularities. This
// value can be used for aligning both physical memory allocations and for
// reserving virtual memory ranges.
size_t GetLCMGranularity(
    const sycl::device &Dev, const sycl::context &Ctx,
    syclext::granularity_mode Gm = syclext::granularity_mode::recommended) {
  size_t CtxGranularity = syclext::get_mem_granularity(Ctx, Gm);
  size_t DevGranularity = syclext::get_mem_granularity(Dev, Ctx, Gm);

  size_t GCD = CtxGranularity;
  size_t Rem = DevGranularity % GCD;
  while (Rem != 0) {
    std::swap(GCD, Rem);
    Rem %= GCD;
  }
  return (DevGranularity / GCD) * CtxGranularity;
}

size_t GetAlignedByteSize(const size_t UnalignedBytes,
                          const size_t AligmentGranularity) {
  return ((UnalignedBytes + AligmentGranularity - 1) / AligmentGranularity) *
         AligmentGranularity;
}

bool check_for_42(std::vector<int> &vec, int ref_result = 42) {
  return vec[42] == ref_result;
}

int main() {
  // Get all available devices
  auto devices = sycl::device::get_devices();

  // Filter out GPU devices
  std::vector<sycl::device> gpuDevices;
  for (const auto &dev : devices) {
    if (dev.is_gpu()) {
      gpuDevices.push_back(dev);
    }
  }

  // Check if we have at least two GPU devices
  if (gpuDevices.size() < 2) {
    std::cerr << "Less than two GPU devices found." << std::endl;
    return 1;
  }

  // Create contexts for the first two GPU devices
  auto dev1 = gpuDevices[0];
  auto dev2 = gpuDevices[1];
  sycl::context context1_d1(dev1);
  sycl::context context2_d1(dev1);
  sycl::context context_d2(dev2);

  sycl::queue Q1_d1(context1_d1, dev1);
  sycl::queue Q2_d1(context2_d1, dev1);
  sycl::queue Q1_d2(context_d2, dev2);

  constexpr size_t NumberOfElements = 1000;
  size_t BytesRequired = NumberOfElements * sizeof(int);
  size_t UsedGranularity = GetLCMGranularity(dev1, context2_d1);
  size_t AlignedByteSize =
      ((BytesRequired + UsedGranularity - 1) / UsedGranularity) *
      UsedGranularity;
  printf("UsedGranularity: %zu\n", UsedGranularity);
  printf("AlignedByteSize: %zu\n", AlignedByteSize);

  syclext::physical_mem NewPhysicalMem{dev1, context2_d1, AlignedByteSize};

  uintptr_t VirtualMemoryPtr =
      syclext::reserve_virtual_mem(0, AlignedByteSize, context2_d1);

  void *MappedPtr =
      NewPhysicalMem.map(VirtualMemoryPtr, AlignedByteSize,
                         syclext::address_access_mode::read_write);

  int *DataPtr = reinterpret_cast<int *>(MappedPtr);
  printf("DataPtr: %p\n", DataPtr);

  std::vector<int> ResultHostData(NumberOfElements);
  constexpr int ExpectedValueAfterFill = 42;

  {
    // Normal case, same device, same context
    sycl::buffer<int> CheckBuffer(ResultHostData);
    Q2_d1.fill(DataPtr, ExpectedValueAfterFill, NumberOfElements)
        .wait_and_throw();
    Q2_d1.submit([&](sycl::handler &Handle) {
      sycl::accessor A(CheckBuffer, Handle, sycl::write_only);
      Handle.parallel_for(NumberOfElements,
                          [=](sycl::id<1> Idx) { A[Idx] = DataPtr[Idx]; });
    });
    Q2_d1.wait();
  }
  assert(check_for_42(ResultHostData));
  ResultHostData = std::vector<int>(NumberOfElements);
  Q2_d1.fill(DataPtr, 0, NumberOfElements).wait_and_throw();
  assert(check_for_42(ResultHostData, 0));

  {
    // !!! Same device, different context !!!
    sycl::buffer<int> CheckBuffer(ResultHostData);
    Q1_d1.fill(DataPtr, ExpectedValueAfterFill, NumberOfElements)
        .wait_and_throw();
    Q1_d1.submit([&](sycl::handler &Handle) {
      sycl::accessor A(CheckBuffer, Handle, sycl::write_only);
      Handle.parallel_for(NumberOfElements,
                          [=](sycl::id<1> Idx) { A[Idx] = DataPtr[Idx]; });
    });
    Q1_d1.wait();
  }
  assert(check_for_42(ResultHostData));
  ResultHostData = std::vector<int>(NumberOfElements);
  Q1_d1.fill(DataPtr, 0, NumberOfElements).wait_and_throw();
  assert(check_for_42(ResultHostData, 0));

  {
    // !!! Different device, different context !!!
    sycl::buffer<int> CheckBuffer(ResultHostData);
    Q1_d2.fill(DataPtr, ExpectedValueAfterFill, NumberOfElements)
        .wait_and_throw();
    Q1_d2.submit([&](sycl::handler &Handle) {
      sycl::accessor A(CheckBuffer, Handle, sycl::write_only);
      Handle.parallel_for(NumberOfElements,
                          [=](sycl::id<1> Idx) { A[Idx] = DataPtr[Idx]; });
    });
    Q1_d2.wait();
  }
  assert(check_for_42(ResultHostData));

  return 0;
}
