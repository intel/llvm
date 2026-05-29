// REQUIRES: aspect-ext_oneapi_virtual_mem && aspect-ext_oneapi_ipc_physical_memory

// UNSUPPORTED: level_zero && windows
// UNSUPPORTED-TRACKER: https://github.com/intel/llvm/issues/0000

// DEFINE: %{cpp20} = %if cl_options %{/clang:-std=c++20%} %else %{-std=c++20%}

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// RUN: %{build} -DUSE_VIEW %{cpp20} -o %t.view.out
// RUN: %{run} %t.view.out

// Tests cross-process IPC for physical_mem objects using the no-fd pattern:
// the spawner creates a physical_mem with enable_ipc and serializes the IPC
// handle to a file WITHOUT mapping the memory first.  The consumer opens the
// physical_mem handle (zeMemOpenIpcHandle establishes the virtual mapping
// directly), writes a known data pattern, then exits.  The spawner then maps
// the same physical memory, reads it, and verifies the consumer's writes.
//
// No file-descriptor passing is needed: zeMemGetIpcHandleWithProperties called
// with a ze_physical_mem_handle_t produces an opaque 64-byte blob that can be
// serialized to a plain file and reconstructed in any process without
// SCM_RIGHTS transfer.  The physical_mem destructor on the consumer side calls
// zeMemCloseIpcHandle to release the IPC mapping.

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/ipc_memory.hpp>
#include <sycl/ext/oneapi/virtual_mem/physical_mem.hpp>
#include <sycl/ext/oneapi/virtual_mem/virtual_mem.hpp>

#include <cassert>
#include <cstdlib>
#include <fstream>
#include <iostream>

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

  std::cout << "[Spawner] Device: " << Dev.get_info<sycl::info::device::name>()
            << "\n";
  std::cout << "[Spawner] Creating physical_mem (" << N
            << " ints) with IPC support...\n";

  const size_t AlignedByteSize =
      alignUp(N * sizeof(int), getGranularity(Dev, Ctx));

  // Create a physical_mem that can be shared via IPC.  No virtual mapping is
  // needed before exporting: zeMemGetIpcHandleWithProperties accepts a
  // ze_physical_mem_handle_t directly and returns an opaque 64-byte blob that
  // requires no fd transfer and can be written to a plain file.
  syclexp::physical_mem PhysMem{Dev, Ctx, AlignedByteSize,
                                syclexp::properties{syclexp::enable_ipc{}}};

  {
    // Obtain the IPC handle and serialize it together with AlignedByteSize.
    std::cout << "[Spawner] Exporting IPC handle to '" << CommsFile << "'...\n";
    syclipc::handle Handle = [&]() {
      try {
        return syclipc::physical_memory::get(PhysMem);
      } catch (const sycl::exception &E) {
        if (E.code() == sycl::errc::feature_not_supported) {
          std::cout << "[Spawner] SKIPPED: " << E.what() << "\n";
          std::exit(0);
        }
        throw;
      }
    }();

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
      return 1;
    }
    FS.write(reinterpret_cast<const char *>(&AlignedByteSize), sizeof(size_t));
    FS.write(reinterpret_cast<const char *>(&HandleDataSize), sizeof(size_t));
    FS.write(reinterpret_cast<const char *>(HandleData.data()), HandleDataSize);
    if (!FS.good()) {
      std::cerr << "[Spawner] Failed to write comms file '" << CommsFile
                << "'\n";
      return 1;
    }
    FS.close();

    // Spawn the consumer process; it opens the physical_mem via the IPC handle
    // and writes a data pattern into the device memory.
    std::string Cmd = std::string{argv[0]} + " 1";
    std::cout << "[Spawner] Spawning consumer: " << Cmd << "\n";
    int Status = std::system(Cmd.c_str());
    if (Status != 0) {
      std::cerr << "[Spawner] Consumer failed with status " << Status << "\n";
      std::remove(CommsFile);
      return 1;
    }

    // Release the IPC handle.
    syclipc::physical_memory::put(Handle, Ctx);
  }

  // After the consumer exits, map the physical memory and verify that the
  // consumer's writes are visible.  This confirms that the IPC-opened mapping
  // in the consumer process and the zeVirtualMemMap-based mapping here both
  // back the same underlying physical memory.
  std::cout << "[Spawner] Mapping " << AlignedByteSize
            << " bytes to verify consumer's writes...\n";
  uintptr_t VAddr = syclexp::reserve_virtual_mem(AlignedByteSize, Ctx);
  int *DataPtr = reinterpret_cast<int *>(PhysMem.map(
      VAddr, AlignedByteSize, syclexp::address_access_mode::read_write));

  // Verify consumer wrote [N, N+1, ..., 2N-1].
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

  // Open the physical_mem from the IPC handle.  zeMemOpenIpcHandle establishes
  // a virtual mapping over the exporter's physical memory directly; no separate
  // reserve_virtual_mem call is needed.  The PhysMem destructor calls
  // zeMemCloseIpcHandle to release the mapping.
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

  // For IPC-opened handles, zeMemOpenIpcHandle already established the virtual
  // mapping.  physical_mem::map() returns the driver-chosen virtual address.
  std::cout << "[Consumer] Obtaining IPC virtual address via map()...\n";
  int *DataPtr = reinterpret_cast<int *>(PhysMem.map(
      0, AlignedByteSize, syclexp::address_access_mode::read_write));

  // Write [N, N+1, ..., 2N-1] into the device memory.  The spawner will map
  // the same physical memory after we exit and verify these values.
  std::cout << "[Consumer] Writing data [" << N << ".." << 2 * N - 1
            << "]...\n";
  Q.parallel_for(N, [=](sycl::item<1> I) {
     DataPtr[I] = static_cast<int>(N + I.get_linear_id());
   }).wait();

  // Read back from device to host and verify the write succeeded in this
  // process before closing the handle.  This mirrors the L0 CTS which also
  // validates the in-process write prior to zeMemCloseIpcHandle.
  std::cout << "[Consumer] Verifying in-process write succeeded...\n";
  int Verify[N] = {0};
  Q.copy(DataPtr, Verify, N).wait();
  int Failures = 0;
  for (size_t I = 0; I < N; ++I) {
    const int Expected = static_cast<int>(N + I);
    if (Verify[I] != Expected) {
      ++Failures;
      std::cout << "[Consumer] MISMATCH at [" << I << "]: got " << Verify[I]
                << ", expected " << Expected << "\n";
    }
  }

  // Cleanup: PhysMem destructor calls zeMemCloseIpcHandle to unmap and release
  // the IPC mapping; no explicit unmap() or free_virtual_mem() is needed.
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
