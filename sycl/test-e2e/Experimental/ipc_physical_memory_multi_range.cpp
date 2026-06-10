// REQUIRES: aspect-ext_oneapi_virtual_mem && aspect-ext_oneapi_ipc_physical_memory

// UNSUPPORTED: level_zero && windows
// UNSUPPORTED-TRACKER: https://github.com/intel/llvm/issues/0000

// DEFINE: %{cpp20} = %if cl_options %{/clang:-std=c++20%} %else %{-std=c++20%}

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// RUN: %{build} -DUSE_VIEW %{cpp20} -o %t.view.out
// RUN: %{run} %t.view.out

// Tests cross-process IPC for multiple physical_mem objects: the spawner
// creates 3 separate physical_mem ranges and exports all 3 IPC handles to a
// comms file, then spawns a consumer.  The consumer opens all 3 handles and
// maps them into one contiguous virtual address space.  It then fills the
// entire contiguous range with a known pattern [0, 1, ..., 3*N-1] and exits.
// The spawner maps each physical_mem range separately, reads back the data,
// and verifies that each range contains its expected slice of the pattern.
//
// Cross-process fd sharing uses pidfd_getfd(2) (Linux 5.6+) internally.
// The spawner must make itself ptrace-accessible so the consumer can copy
// the DMA-BUF fds via pidfd_getfd.

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/ipc_memory.hpp>
#include <sycl/ext/oneapi/virtual_mem/physical_mem.hpp>
#include <sycl/ext/oneapi/virtual_mem/virtual_mem.hpp>

#include <cassert>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <string_view>
#include <vector>

#if defined(__linux__)
#include <linux/prctl.h>
#include <sys/prctl.h>
#include <unistd.h>
#endif // defined(__linux__)

namespace syclexp = sycl::ext::oneapi::experimental;
namespace syclipc = sycl::ext::oneapi::experimental::ipc;

constexpr size_t N = 32;        // ints per physical_mem range
constexpr size_t NumRanges = 3; // number of separate physical_mem ranges
constexpr const char *CommsFile = "ipc_phys_mem_multi_range_comms.txt";

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

#if defined(__linux__)
  // Allow any process (the spawned consumer) to copy our DMA-BUF fds via
  // pidfd_getfd(2), which internally uses PTRACE_MODE_ATTACH permissions.
  // Without this, pidfd_getfd fails with EPERM under ptrace_scope=1.
  prctl(PR_SET_PTRACER, PR_SET_PTRACER_ANY);
#endif // defined(__linux__)

  const size_t AlignedByteSize =
      alignUp(N * sizeof(int), getGranularity(Dev, Ctx));

  std::cout << "[Spawner] Creating " << NumRanges << " physical_mem ranges ("
            << N << " ints, " << AlignedByteSize << " bytes each)...\n";

  // Create NumRanges physical_mem objects with IPC support.
  std::vector<syclexp::physical_mem> PhysMemVec;
  PhysMemVec.reserve(NumRanges);
  try {
    for (size_t R = 0; R < NumRanges; ++R)
      PhysMemVec.emplace_back(Dev, Ctx, AlignedByteSize,
                              syclexp::properties{syclexp::enable_ipc{}});
  } catch (const sycl::exception &E) {
    if (std::string_view{E.what()}.find("UNSUPPORTED_FEATURE") !=
        std::string_view::npos) {
      std::cout << "[Spawner] Skipping: driver does not support IPC physical "
                   "memory ("
                << E.what() << ")\n";
      return 0;
    }
    throw;
  }

  // Map all ranges into a single contiguous VA in the spawner now, before
  // spawning the consumer.  This lets the spawner verify consumer writes by
  // simply reading from the pre-existing mappings — no additional
  // reserve_virtual_mem / map calls are needed after the consumer exits,
  // avoiding extra L0 VA reservation pressure under parallel test execution.
  const size_t TotalByteSize = NumRanges * AlignedByteSize;
  std::cout << "[Spawner] Reserving " << TotalByteSize
            << " bytes of virtual address space...\n";
  uintptr_t VBase = syclexp::reserve_virtual_mem(TotalByteSize, Ctx);
  std::vector<int *> DataPtrs(NumRanges);
  for (size_t R = 0; R < NumRanges; ++R) {
    uintptr_t SlotAddr = VBase + R * AlignedByteSize;
    DataPtrs[R] = reinterpret_cast<int *>(PhysMemVec[R].map(
        SlotAddr, AlignedByteSize, syclexp::address_access_mode::read_write));
  }

  // Obtain an IPC handle for each range.
  std::vector<syclipc::handle> Handles;
  Handles.reserve(NumRanges);
  try {
    for (size_t R = 0; R < NumRanges; ++R)
      Handles.push_back(syclipc::physical_memory::get(PhysMemVec[R]));
  } catch (const sycl::exception &E) {
    if (std::string_view{E.what()}.find("UNSUPPORTED_FEATURE") !=
        std::string_view::npos) {
      std::cout << "[Spawner] Skipping: driver does not support IPC physical "
                   "memory export ("
                << E.what() << ")\n";
      for (size_t R = 0; R < Handles.size(); ++R)
        syclipc::physical_memory::put(Handles[R], Ctx);
      for (size_t R = 0; R < NumRanges; ++R)
        syclexp::unmap(DataPtrs[R], AlignedByteSize, Ctx);
      syclexp::free_virtual_mem(VBase, TotalByteSize, Ctx);
      return 0;
    }
    throw;
  }

  // Serialize all handles to the comms file:
  //   NumRanges (size_t)
  //   AlignedByteSize (size_t)
  //   for each range: HandleDataSize (size_t) + HandleData bytes
  std::cout << "[Spawner] Exporting " << NumRanges << " IPC handles to '"
            << CommsFile << "'...\n";
  {
    std::fstream FS(CommsFile, std::ios_base::out | std::ios_base::binary);
    if (!FS.is_open()) {
      std::cerr << "[Spawner] Failed to open comms file '" << CommsFile
                << "' for writing\n";
      for (size_t R = 0; R < NumRanges; ++R)
        syclipc::physical_memory::put(Handles[R], Ctx);
      for (size_t R = 0; R < NumRanges; ++R)
        syclexp::unmap(DataPtrs[R], AlignedByteSize, Ctx);
      syclexp::free_virtual_mem(VBase, TotalByteSize, Ctx);
      return 1;
    }
    FS.write(reinterpret_cast<const char *>(&NumRanges), sizeof(size_t));
    FS.write(reinterpret_cast<const char *>(&AlignedByteSize), sizeof(size_t));
    for (size_t R = 0; R < NumRanges; ++R) {
#ifdef USE_VIEW
      syclipc::handle_data_view_t HandleData = Handles[R].data_view();
#else
      syclipc::handle_data_t HandleData = Handles[R].data();
#endif
      size_t HandleDataSize = HandleData.size();
      FS.write(reinterpret_cast<const char *>(&HandleDataSize), sizeof(size_t));
      FS.write(reinterpret_cast<const char *>(HandleData.data()),
               HandleDataSize);
    }
    if (!FS.good()) {
      std::cerr << "[Spawner] Failed to write comms file '" << CommsFile
                << "'\n";
      for (size_t R = 0; R < NumRanges; ++R)
        syclipc::physical_memory::put(Handles[R], Ctx);
      for (size_t R = 0; R < NumRanges; ++R)
        syclexp::unmap(DataPtrs[R], AlignedByteSize, Ctx);
      syclexp::free_virtual_mem(VBase, TotalByteSize, Ctx);
      return 1;
    }
  }

  // Spawn the consumer; it maps all ranges contiguously and writes the pattern.
  std::string Cmd = std::string{argv[0]} + " 1";
  std::cout << "[Spawner] Spawning consumer: " << Cmd << "\n";
  int Status = std::system(Cmd.c_str());
  if (Status != 0) {
    std::cerr << "[Spawner] Consumer failed with status " << Status << "\n";
    std::remove(CommsFile);
    for (size_t R = 0; R < NumRanges; ++R)
      syclipc::physical_memory::put(Handles[R], Ctx);
    for (size_t R = 0; R < NumRanges; ++R)
      syclexp::unmap(DataPtrs[R], AlignedByteSize, Ctx);
    syclexp::free_virtual_mem(VBase, TotalByteSize, Ctx);
    return 1;
  }

  // Release all IPC handles.
  for (size_t R = 0; R < NumRanges; ++R)
    syclipc::physical_memory::put(Handles[R], Ctx);

  // Verify: read from the pre-existing mappings (no new reserve/map needed).
  // Range R should contain the slice [R*N, R*N+1, ..., R*N+N-1].
  std::cout << "[Spawner] Verifying consumer wrote pattern across " << NumRanges
            << " ranges...\n";
  int TotalFailures = 0;
  for (size_t R = 0; R < NumRanges; ++R) {
    int Read[N] = {0};
    Q.copy(DataPtrs[R], Read, N).wait();
    for (size_t I = 0; I < N; ++I) {
      const int Expected = static_cast<int>(R * N + I);
      if (Read[I] != Expected) {
        ++TotalFailures;
        std::cout << "[Spawner] Range[" << R << "] MISMATCH at [" << I
                  << "]: got " << Read[I] << ", expected " << Expected << "\n";
      }
    }
  }

  // Cleanup virtual address space.
  for (size_t R = 0; R < NumRanges; ++R)
    syclexp::unmap(DataPtrs[R], AlignedByteSize, Ctx);
  syclexp::free_virtual_mem(VBase, TotalByteSize, Ctx);
  std::remove(CommsFile);

  if (TotalFailures == 0)
    std::cout << "[Spawner] PASSED\n";
  else
    std::cout << "[Spawner] FAILED (" << TotalFailures << " mismatches)\n";
  return TotalFailures;
}

int consumer() {
  sycl::queue Q;
  sycl::context Ctx = Q.get_context();
  sycl::device Dev = Q.get_device();

  std::cout << "[Consumer] Device: " << Dev.get_info<sycl::info::device::name>()
            << "\n";
  std::cout << "[Consumer] Reading IPC handles from '" << CommsFile << "'...\n";

  // Read the serialized handles from the comms file.
  std::fstream FS(CommsFile, std::ios_base::in | std::ios_base::binary);
  if (!FS.is_open()) {
    std::cerr << "[Consumer] Failed to open comms file '" << CommsFile << "'\n";
    return 1;
  }
  size_t FileNumRanges = 0;
  FS.read(reinterpret_cast<char *>(&FileNumRanges), sizeof(size_t));
  size_t AlignedByteSize = 0;
  FS.read(reinterpret_cast<char *>(&AlignedByteSize), sizeof(size_t));

  std::vector<std::unique_ptr<std::byte[]>> HandleBytesVec(FileNumRanges);
  std::vector<size_t> HandleSizes(FileNumRanges);
  for (size_t R = 0; R < FileNumRanges; ++R) {
    FS.read(reinterpret_cast<char *>(&HandleSizes[R]), sizeof(size_t));
    HandleBytesVec[R].reset(new std::byte[HandleSizes[R]]);
    FS.read(reinterpret_cast<char *>(HandleBytesVec[R].get()), HandleSizes[R]);
  }
  if (!FS.good()) {
    std::cerr << "[Consumer] Failed to read comms file '" << CommsFile << "'\n";
    return 1;
  }
  FS.close();

  // Open all physical_mem handles.
  std::cout << "[Consumer] Opening " << FileNumRanges
            << " physical_mem handles...\n";
  std::vector<syclexp::physical_mem> PhysMemVec;
  PhysMemVec.reserve(FileNumRanges);
  for (size_t R = 0; R < FileNumRanges; ++R) {
#ifdef USE_VIEW
    syclipc::handle_data_view_t HandleData{HandleBytesVec[R].get(),
                                           HandleSizes[R]};
#else
    syclipc::handle_data_t HandleData{HandleBytesVec[R].get(),
                                      HandleBytesVec[R].get() + HandleSizes[R]};
#endif
    PhysMemVec.emplace_back(
        syclipc::physical_memory::open(HandleData, Ctx, Dev));
  }

  // Reserve one contiguous virtual address space large enough for all ranges.
  const size_t TotalByteSize = FileNumRanges * AlignedByteSize;
  std::cout << "[Consumer] Reserving " << TotalByteSize
            << " bytes of contiguous virtual address space for "
            << FileNumRanges << " ranges...\n";
  uintptr_t VBase = syclexp::reserve_virtual_mem(TotalByteSize, Ctx);

  // Map each physical_mem into its consecutive slot in the virtual space:
  //   Range 0 → VBase + 0
  //   Range 1 → VBase + AlignedByteSize
  //   Range 2 → VBase + 2 * AlignedByteSize
  std::cout << "[Consumer] Mapping " << FileNumRanges
            << " ranges into contiguous virtual space...\n";
  std::vector<int *> DataPtrs(FileNumRanges);
  for (size_t R = 0; R < FileNumRanges; ++R) {
    uintptr_t SlotAddr = VBase + R * AlignedByteSize;
    DataPtrs[R] = reinterpret_cast<int *>(PhysMemVec[R].map(
        SlotAddr, AlignedByteSize, syclexp::address_access_mode::read_write));
  }

  // The first range's mapped pointer is the base of the contiguous allocation.
  int *ContiguousPtr = DataPtrs[0];

  // Each physical range occupies AlignedByteSize bytes in the contiguous VA
  // space.  Write N ints at the start of each range's slot so that range R
  // holds the slice [R*N, R*N+1, ..., R*N+N-1].  The stride between the
  // starts of consecutive range slots in the int* view is IntsPerRange.
  const size_t IntsPerRange = AlignedByteSize / sizeof(int);
  std::cout << "[Consumer] Writing pattern across " << FileNumRanges
            << " ranges (N=" << N << " ints at start of each "
            << AlignedByteSize << "-byte slot)...\n";
  Q.parallel_for(sycl::range<2>{FileNumRanges, N}, [=](sycl::item<2> I) {
     size_t R = I[0];
     size_t J = I[1];
     ContiguousPtr[R * IntsPerRange + J] = static_cast<int>(R * N + J);
   }).wait();

  // Verify the write in-process: read back N ints from the start of each
  // range slot and confirm they match the expected slice.
  std::cout << "[Consumer] Verifying in-process write...\n";
  int Failures = 0;
  for (size_t R = 0; R < FileNumRanges; ++R) {
    std::vector<int> Verify(N, 0);
    Q.copy(ContiguousPtr + R * IntsPerRange, Verify.data(), N).wait();
    for (size_t J = 0; J < N; ++J) {
      const int Expected = static_cast<int>(R * N + J);
      if (Verify[J] != Expected) {
        ++Failures;
        std::cout << "[Consumer] Range[" << R << "] MISMATCH at [" << J
                  << "]: got " << Verify[J] << ", expected " << Expected
                  << "\n";
      }
    }
  }

  // Unmap each individual slot, then release the contiguous reservation.
  for (size_t R = 0; R < FileNumRanges; ++R)
    syclexp::unmap(DataPtrs[R], AlignedByteSize, Ctx);
  syclexp::free_virtual_mem(VBase, TotalByteSize, Ctx);

  if (Failures == 0)
    std::cout << "[Consumer] PASSED\n";
  else
    std::cout << "[Consumer] FAILED (" << Failures << " mismatches)\n";
  return Failures;
}

int main(int argc, char *argv[]) {
  if (argc == 1)
    std::cout << "=== ipc_physical_memory_multi_range test"
#ifdef USE_VIEW
              << " (USE_VIEW)"
#endif
              << " ===\n";
  return argc == 1 ? spawner(argc, argv) : consumer();
}
