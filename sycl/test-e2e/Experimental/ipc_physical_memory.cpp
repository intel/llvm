// REQUIRES: aspect-ext_oneapi_virtual_mem && aspect-ext_oneapi_ipc_physical_memory

// UNSUPPORTED: level_zero && windows
// UNSUPPORTED-TRACKER: https://github.com/intel/llvm/issues/0000

// DEFINE: %{cpp20} = %if cl_options %{/clang:-std=c++20%} %else %{-std=c++20%}

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// RUN: %{build} -DUSE_VIEW %{cpp20} -o %t.view.out
// RUN: %{run} %t.view.out

// Tests cross-process IPC for physical_mem objects: the spawner creates a
// physical_mem with enable_ipc, maps it, writes data, serializes the IPC
// handle to a file, and spawns a consumer.  The consumer opens the physical
// mem handle, verifies the data, writes new data, then exits.  The spawner
// verifies the consumer's writes.
//
// Cross-process fd sharing uses pidfd_getfd(2) (Linux 5.6+) internally.
// The spawner must make itself ptrace-accessible so the consumer can copy
// the DMA-BUF fd via pidfd_getfd.

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/ipc_memory.hpp>
#include <sycl/ext/oneapi/virtual_mem/physical_mem.hpp>
#include <sycl/ext/oneapi/virtual_mem/virtual_mem.hpp>

#include <cassert>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string_view>

#if defined(__linux__)
#include <linux/prctl.h>
#include <sys/prctl.h>
#include <unistd.h>
#endif // defined(__linux__)

namespace syclexp = sycl::ext::oneapi::experimental;
namespace syclipc = sycl::ext::oneapi::experimental::ipc;

constexpr size_t N = 32;
constexpr const char *CommsFile = "ipc_phys_mem_comms.txt";

// Return the smallest multiple of Granularity that is >= Bytes.
static size_t alignUp(size_t Bytes, size_t Granularity) {
  return ((Bytes + Granularity - 1) / Granularity) * Granularity;
}

// Return the granularity to use for both physical and virtual allocations.
static size_t getGranularity(const sycl::device &Dev,
                             const sycl::context &Ctx) {
  size_t CtxGran = syclexp::get_mem_granularity(Ctx);
  size_t DevGran = syclexp::get_mem_granularity(Dev, Ctx);
  // Use the LCM so the size is aligned to both constraints.
  size_t GCD = CtxGran;
  size_t Rem = DevGran % GCD;
  while (Rem != 0) {
    std::swap(GCD, Rem);
    Rem %= GCD;
  }
  return (DevGran / GCD) * CtxGran;
}

int spawner(int argc, char *argv[]) {
  assert(argc == 1);
  sycl::queue Q;
  sycl::context Ctx = Q.get_context();
  sycl::device Dev = Q.get_device();

  if (!Dev.has(sycl::aspect::ext_oneapi_ipc_physical_memory)) {
    std::cout << "[Spawner] Skipping: device does not support "
                 "aspect::ext_oneapi_ipc_physical_memory\n";
    return 0;
  }

  std::cout << "[Spawner] Device: " << Dev.get_info<sycl::info::device::name>()
            << "\n";
  std::cout << "[Spawner] Creating physical_mem (" << N
            << " ints) with IPC support...\n";

#if defined(__linux__)
  // Allow any process (the spawned consumer) to copy our DMA-BUF fd via
  // pidfd_getfd(2), which internally uses PTRACE_MODE_ATTACH permissions.
  // Without this, pidfd_getfd fails with EPERM under ptrace_scope=1.
  prctl(PR_SET_PTRACER, PR_SET_PTRACER_ANY);
#endif // defined(__linux__)

  const size_t AlignedByteSize =
      alignUp(N * sizeof(int), getGranularity(Dev, Ctx));

  // Create a physical_mem that can be shared via IPC.
  syclexp::physical_mem PhysMem = [&]() -> syclexp::physical_mem {
    try {
      return syclexp::physical_mem{Dev, Ctx, AlignedByteSize,
                                   syclexp::properties{syclexp::enable_ipc{}}};
    } catch (const sycl::exception &E) {
      if (std::string_view{E.what()}.find("UNSUPPORTED_FEATURE") !=
          std::string_view::npos) {
        std::cout << "[Spawner] Skipping: driver does not support IPC physical "
                     "memory ("
                  << E.what() << ")\n";
        std::exit(0);
      }
      throw;
    }
  }();

  // Reserve virtual address space and map the physical memory into it.
  std::cout << "[Spawner] Mapping " << AlignedByteSize
            << " bytes of virtual address space...\n";
  uintptr_t VAddr = syclexp::reserve_virtual_mem(AlignedByteSize, Ctx);
  int *DataPtr = reinterpret_cast<int *>(PhysMem.map(
      VAddr, AlignedByteSize, syclexp::address_access_mode::read_write));

  // Initialize: write [0, 1, ..., N-1] into the mapped memory.
  std::cout << "[Spawner] Writing initial data [0.." << N - 1 << "]...\n";
  Q.parallel_for(N, [=](sycl::item<1> I) {
     DataPtr[I] = static_cast<int>(I.get_linear_id());
   }).wait();

  try {
    // Obtain the IPC handle and serialize it together with AlignedByteSize.
    std::cout << "[Spawner] Exporting IPC handle to '" << CommsFile << "'...\n";
    syclipc::handle Handle = syclipc::physical_memory::get(PhysMem);

#ifdef USE_VIEW
    syclipc::handle_data_view_t HandleData = Handle.data_view();
#else
    syclipc::handle_data_t HandleData = Handle.data();
#endif

    size_t HandleDataSize = HandleData.size();
    std::fstream FS(CommsFile, std::ios_base::out | std::ios_base::binary);
    if (!FS.is_open()) {
      std::cerr << "[Spawner] Failed to open comms file '" << CommsFile
                << "' for writing\n";
      syclexp::unmap(DataPtr, AlignedByteSize, Ctx);
      syclexp::free_virtual_mem(VAddr, AlignedByteSize, Ctx);
      return 1;
    }
    FS.write(reinterpret_cast<const char *>(&AlignedByteSize), sizeof(size_t));
    FS.write(reinterpret_cast<const char *>(&HandleDataSize), sizeof(size_t));
    FS.write(reinterpret_cast<const char *>(HandleData.data()), HandleDataSize);
    if (!FS.good()) {
      std::cerr << "[Spawner] Failed to write comms file '" << CommsFile
                << "'\n";
      syclexp::unmap(DataPtr, AlignedByteSize, Ctx);
      syclexp::free_virtual_mem(VAddr, AlignedByteSize, Ctx);
      return 1;
    }
    FS.close();

    // Spawn the consumer process; it reads the comms file, opens the
    // physical_mem, verifies the spawner's data, and writes new values.
    std::string Cmd = std::string{argv[0]} + " 1";
    std::cout << "[Spawner] Spawning consumer: " << Cmd << "\n";
    int Status = std::system(Cmd.c_str());
    if (Status != 0) {
      std::cerr << "[Spawner] Consumer failed with status " << Status << "\n";
      std::remove(CommsFile);
      syclexp::unmap(DataPtr, AlignedByteSize, Ctx);
      syclexp::free_virtual_mem(VAddr, AlignedByteSize, Ctx);
      return 1;
    }

    // Release the IPC handle (closes the exported fd).
    syclipc::physical_memory::put(Handle, Ctx);
  } catch (const sycl::exception &E) {
    if (std::string_view{E.what()}.find("UNSUPPORTED_FEATURE") !=
        std::string_view::npos) {
      std::cout << "[Spawner] Skipping: driver does not support IPC physical "
                   "memory export ("
                << E.what() << ")\n";
      syclexp::unmap(DataPtr, AlignedByteSize, Ctx);
      syclexp::free_virtual_mem(VAddr, AlignedByteSize, Ctx);
      return 0;
    }
    throw;
  }

  // After consumer exits, verify it wrote [N, N+1, ..., 2N-1].
  std::cout << "[Spawner] Verifying consumer wrote [" << N << ".." << 2 * N - 1
            << "]...\n";
  int Failures = 0;
  int Read[N] = {0};
  Q.copy(DataPtr, Read, N).wait();
  for (size_t I = 0; I < N; ++I) {
    const int Expected = static_cast<int>(N + I);
    if (Read[I] != Expected) {
      ++Failures;
      std::cout << "[Spawner] MISMATCH at [" << I << "]: got " << Read[I]
                << ", expected " << Expected << "\n";
    }
  }

  // Cleanup.
  syclexp::unmap(DataPtr, AlignedByteSize, Ctx);
  syclexp::free_virtual_mem(VAddr, AlignedByteSize, Ctx);
  std::remove(CommsFile);

  if (Failures == 0)
    std::cout << "[Spawner] PASSED\n";
  else
    std::cout << "[Spawner] FAILED (" << Failures << " mismatches)\n";
  return Failures;
}

int consumer() {
  sycl::queue Q;
  sycl::context Ctx = Q.get_context();
  sycl::device Dev = Q.get_device();

  std::cout << "[Consumer] Device: " << Dev.get_info<sycl::info::device::name>()
            << "\n";
  std::cout << "[Consumer] Reading IPC handle from '" << CommsFile << "'...\n";

  // Read the serialized handle from the comms file.
  std::fstream FS(CommsFile, std::ios_base::in | std::ios_base::binary);
  if (!FS.is_open()) {
    std::cerr << "[Consumer] Failed to open comms file '" << CommsFile << "'\n";
    return 1;
  }
  size_t AlignedByteSize = 0;
  FS.read(reinterpret_cast<char *>(&AlignedByteSize), sizeof(size_t));
  size_t HandleSize = 0;
  FS.read(reinterpret_cast<char *>(&HandleSize), sizeof(size_t));
  std::unique_ptr<std::byte[]> HandleBytes{new std::byte[HandleSize]};
  FS.read(reinterpret_cast<char *>(HandleBytes.get()), HandleSize);
  if (!FS.good()) {
    std::cerr << "[Consumer] Failed to read comms file '" << CommsFile << "'\n";
    return 1;
  }
  FS.close();

  // Open the physical_mem from the IPC handle.  The resulting object will
  // call urIPCClosePhysMemHandleExp when destroyed.
  std::cout << "[Consumer] Opening physical_mem from IPC handle (" << HandleSize
            << " bytes)...\n";
#ifdef USE_VIEW
  syclipc::handle_data_view_t HandleData{HandleBytes.get(), HandleSize};
#else
  syclipc::handle_data_t HandleData{HandleBytes.get(),
                                    HandleBytes.get() + HandleSize};
#endif

  syclexp::physical_mem PhysMem =
      syclipc::physical_memory::open(HandleData, Ctx, Dev);

  // Map the opened physical_mem at a fresh virtual address.
  std::cout << "[Consumer] Mapping " << AlignedByteSize
            << " bytes of virtual address space...\n";
  uintptr_t VAddr = syclexp::reserve_virtual_mem(AlignedByteSize, Ctx);
  int *DataPtr = reinterpret_cast<int *>(PhysMem.map(
      VAddr, AlignedByteSize, syclexp::address_access_mode::read_write));

  // Verify the spawner wrote [0, 1, ..., N-1].
  std::cout << "[Consumer] Verifying spawner wrote [0.." << N - 1 << "]...\n";
  int Failures = 0;
  int Read[N] = {0};
  Q.copy(DataPtr, Read, N).wait();
  for (size_t I = 0; I < N; ++I) {
    const int Expected = static_cast<int>(I);
    if (Read[I] != Expected) {
      ++Failures;
      std::cout << "[Consumer] MISMATCH at [" << I << "]: got " << Read[I]
                << ", expected " << Expected << "\n";
    }
  }

  // Write [N, N+1, ..., 2N-1] so the spawner can verify cross-process writes.
  std::cout << "[Consumer] Writing new data [" << N << ".." << 2 * N - 1
            << "]...\n";
  Q.parallel_for(N, [=](sycl::item<1> I) {
     DataPtr[I] = static_cast<int>(N + I.get_linear_id());
   }).wait();

  // Cleanup virtual-address resources; PhysMem destructor closes the handle.
  syclexp::unmap(DataPtr, AlignedByteSize, Ctx);
  syclexp::free_virtual_mem(VAddr, AlignedByteSize, Ctx);

  if (Failures == 0)
    std::cout << "[Consumer] PASSED\n";
  else
    std::cout << "[Consumer] FAILED (" << Failures << " mismatches)\n";
  return Failures;
}

int main(int argc, char *argv[]) {
  if (argc == 1)
    std::cout << "=== ipc_physical_memory test"
#ifdef USE_VIEW
              << " (USE_VIEW)"
#endif
              << " ===\n";
  return argc == 1 ? spawner(argc, argv) : consumer();
}
