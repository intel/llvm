// REQUIRES: aspect-ext_oneapi_virtual_mem
// REQUIRES: aspect-ext_oneapi_ipc_physical_memory

// UNSUPPORTED: level_zero && windows
// UNSUPPORTED-TRACKER: UMFW-348

// DEFINE: %{cpp20} = %if cl_options %{/clang:-std=c++20%} %else %{-std=c++20%}

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// RUN: %{build} -DUSE_VIEW %{cpp20} -o %t.view.out
// RUN: %{run} %t.view.out

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/ipc_physical_memory.hpp>
#include <sycl/ext/oneapi/virtual_mem/physical_mem.hpp>
#include <sycl/ext/oneapi/virtual_mem/virtual_mem.hpp>

#include <cstdio>
#include <cstdlib>
#include <fstream>

#if defined(__linux__)
#include <linux/prctl.h>
#include <sys/prctl.h>
#include <unistd.h>
#endif // defined(__linux__)

#include "../VirtualMem/helpers.hpp"

namespace syclexp = sycl::ext::oneapi::experimental;

constexpr size_t N = 32;
constexpr const char *CommsFile = "ipc_physical_mem_comms.txt";

int spawner(int argc, char *argv[]) {
  assert(argc == 1);
  sycl::queue Q;

#if defined(__linux__)
  // UR currently requires ptrace permissions to be set for the spawner. As
  // such we need to set it until this limitation has been addressed.
  // https://github.com/oneapi-src/unified-memory-framework/tree/main?tab=readme-ov-file#level-zero-memory-provider
  if (Q.get_backend() == sycl::backend::ext_oneapi_level_zero &&
      prctl(PR_SET_PTRACER, getppid()) == -1) {
    std::cout << "Failed to set ptracer permissions!" << std::endl;
    return 1;
  }
#endif // defined(__linux__)

  sycl::device Device = Q.get_device();
  sycl::context Context = Q.get_context();

  // Calculate required size for virtual memory
  size_t BytesRequired = N * sizeof(int);
  size_t UsedGranularity = GetLCMGranularity(Device, Context);
  size_t AlignedByteSize = GetAlignedByteSize(BytesRequired, UsedGranularity);

  // Reserve virtual memory
  uintptr_t VirtualMemoryPtr =
      syclexp::reserve_virtual_mem(0, AlignedByteSize, Context);

  // Create physical memory with IPC enabled
  syclexp::properties PropList{syclexp::enable_ipc};
  syclexp::physical_mem PhysMem{Device, Context, AlignedByteSize, PropList};

  // Map physical memory to virtual address
  void *MappedPtr = PhysMem.map(VirtualMemoryPtr, AlignedByteSize,
                                syclexp::address_access_mode::read_write);
  int *DataPtr = static_cast<int *>(MappedPtr);

  // Initialize data on device
  Q.parallel_for(N, [=](sycl::item<1> I) {
     DataPtr[I] = static_cast<int>(I.get_linear_id());
   }).wait();

  // Get the IPC handle
  syclexp::ipc::handle IPCHandle = syclexp::ipc::physical_memory::get(PhysMem);

  {
    // Write handle data to file.
  #ifdef USE_VIEW
    syclexp::ipc::handle_data_view_t HandleData = IPCHandle.data_view();
  #else
    syclexp::ipc::handle_data_t HandleData = IPCHandle.data();
#endif
    size_t HandleDataSize = HandleData.size();
    std::fstream FS(CommsFile, std::ios_base::out | std::ios_base::binary);
    FS.write(reinterpret_cast<const char *>(&HandleDataSize), sizeof(size_t));
    FS.write(reinterpret_cast<const char *>(HandleData.data()), HandleDataSize);
  }

  // Spawn other process with an argument.
  std::string Cmd = std::string{argv[0]} + " 1";
  std::cout << "Spawning: " << Cmd << std::endl;
  std::system(Cmd.c_str());

  // Verify that consumer modified the data
  int Failures = 0;
  int Read[N] = {0};
  Q.copy(DataPtr, Read, N).wait();
  for (size_t I = 0; I < N; ++I) {
    int Expected = static_cast<int>(N - I);
    if (Read[I] != Expected) {
      ++Failures;
      std::cout << "Failed from spawner: Result at " << I
                << " unexpected: " << Read[I] << " != " << Expected
                << std::endl;
    }
  }

  // Close the IPC handle
  syclexp::ipc::physical_memory::put(IPCHandle, Context);

  // Cleanup
  syclexp::unmap(MappedPtr, AlignedByteSize, Context);
  syclexp::free_virtual_mem(VirtualMemoryPtr, AlignedByteSize, Context);

  return Failures;
}

int consumer() {
  sycl::queue Q;
  sycl::device Device = Q.get_device();
  sycl::context Context = Q.get_context();

  // Read the handle data.
  std::fstream FS(CommsFile, std::ios_base::in | std::ios_base::binary);
  size_t HandleSize = 0;
  FS.read(reinterpret_cast<char *>(&HandleSize), sizeof(size_t));
  std::unique_ptr<std::byte[]> HandleData{new std::byte[HandleSize]};
  FS.read(reinterpret_cast<char *>(HandleData.get()), HandleSize);

  // Open IPC handle to get physical_mem object
#ifdef USE_VIEW
  syclexp::ipc::handle_data_view_t HandleDataView{HandleData.get(), HandleSize};
  syclexp::physical_mem PhysMem =
      syclexp::ipc::physical_memory::open(HandleDataView, Context, Device);
#else
  syclexp::ipc::handle_data_t HandleDataVec{HandleData.get(),
                                            HandleData.get() + HandleSize};
  syclexp::physical_mem PhysMem =
      syclexp::ipc::physical_memory::open(HandleDataVec, Context, Device);
#endif

  int Failures = 0;

  // Get required size for virtual memory
  size_t AlignedByteSize = PhysMem.size();

  // Calculate size of the memory allocation to verify if the physical mem
  // object holds a correct value after opening the IPC handle.
  size_t BytesRequiredVerify = N * sizeof(int);
  size_t UsedGranularityVerify = GetLCMGranularity(Device, Context);
  size_t AlignedByteSizeVerify =
      GetAlignedByteSize(BytesRequiredVerify, UsedGranularityVerify);

  if (AlignedByteSize != AlignedByteSizeVerify) {
    ++Failures;
    std::cout
        << "Failed from consumer: The physical memory size does not match."
        << " Expected: " << AlignedByteSizeVerify
        << " Returned from physical mem"
        << " object: " << AlignedByteSize;
  }

  // Reserve virtual memory in consumer process
  uintptr_t VirtualMemoryPtr =
      syclexp::reserve_virtual_mem(0, AlignedByteSize, Context);

  // Map the opened physical memory to consumer's virtual address space
  void *MappedPtr = PhysMem.map(VirtualMemoryPtr, AlignedByteSize,
                                syclexp::address_access_mode::read_write);
  int *DataPtr = static_cast<int *>(MappedPtr);

  // Test the data already in the physical memory.
  int Read[N] = {0};
  Q.copy(DataPtr, Read, N).wait();
  for (size_t I = 0; I < N; ++I) {
    int Expected = static_cast<int>(I);
    if (Read[I] != Expected) {
      ++Failures;
      std::cout << "Failed from consumer: Result at " << I
                << " unexpected: " << Read[I] << " != " << Expected
                << std::endl;
    }
  }

  // Modify the data
  Q.parallel_for(N, [=](sycl::item<1> I) {
     DataPtr[I] = static_cast<int>(N - I.get_linear_id());
   }).wait();

  // Cleanup consumer's mapping
  syclexp::unmap(MappedPtr, AlignedByteSize, Context);
  syclexp::free_virtual_mem(VirtualMemoryPtr, AlignedByteSize, Context);

  return Failures;
}

int main(int argc, char *argv[]) {
  return argc == 1 ? spawner(argc, argv) : consumer();
}
