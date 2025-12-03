// REQUIRES: aspect-usm_device_allocations && aspect-ext_oneapi_ipc_memory

// UNSUPPORTED: level_zero && windows
// UNSUPPORTED-TRACKER: UMFW-348

// DEFINE: %{cpp20} = %if cl_options %{/clang:-std=c++20%} %else %{-std=c++20%}

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// RUN: %{build} -DUSE_VIEW %{cpp20} -o %t.view.out
// RUN: %{run} %t.view.out

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/ipc_memory.hpp>
#include <sycl/usm.hpp>

#include <cstdio>
#include <cstdlib>
#include <fstream>

#if defined(__linux__)
#include <linux/prctl.h>
#include <sys/prctl.h>
#include <unistd.h>
#endif // defined(__linux__)

namespace syclexp = sycl::ext::oneapi::experimental;

constexpr size_t N = 32;
constexpr const char *CommsFile = "ipc_comms.txt";

int spawner(int argc, char *argv[]) {
  assert(argc == 1);
  sycl::queue Q;

#if defined(__linux__)
  // UMF currently requires ptrace permissions to be set for the spawner. As
  // such we need to set it until this limitation has been addressed.
  // https://github.com/oneapi-src/unified-memory-framework/tree/main?tab=readme-ov-file#level-zero-memory-provider
  if (Q.get_backend() == sycl::backend::ext_oneapi_level_zero &&
      prctl(PR_SET_PTRACER, getppid()) == -1) {
    std::cout << "Failed to set ptracer permissions!" << std::endl;
    return 1;
  }
#endif // defined(__linux__)

  int *DataPtr = sycl::malloc_device<int>(N, Q);
  Q.parallel_for(N, [=](sycl::item<1> I) {
     DataPtr[I] = static_cast<int>(I.get_linear_id());
   }).wait();

  {
    // Write handle data to file.
    {
      syclexp::ipc_memory::handle Handle =
          syclexp::ipc_memory::get(DataPtr, Q.get_context());
#ifdef USE_VIEW
      syclexp::ipc_memory::handle_data_view_t HandleData = Handle.data_view();
#else
      syclexp::ipc_memory::handle_data_t HandleData = Handle.data();
#endif
      size_t HandleDataSize = HandleData.size();
      std::fstream FS(CommsFile, std::ios_base::out | std::ios_base::binary);
      FS.write(reinterpret_cast<const char *>(&HandleDataSize), sizeof(size_t));
      FS.write(reinterpret_cast<const char *>(HandleData.data()),
               HandleDataSize);
    }

    // Spawn other process with an argument.
    std::string Cmd = std::string{argv[0]} + " 1";
    std::cout << "Spawning: " << Cmd << std::endl;
    std::system(Cmd.c_str());
  }

  int Failures = 0;
  int Read[N] = {0};
  Q.copy(DataPtr, Read, N).wait();
  for (size_t I = 0; I < N; ++I) {
    if (Read[I] != (N - I)) {
      ++Failures;
      std::cout << "Failed from spawner: Result at " << I
                << " unexpected: " << Read[I] << " != " << (N - I) << std::endl;
    }
  }
  sycl::free(DataPtr, Q);
  return Failures;
}

int consumer() {
  sycl::queue Q;

  // Read the handle data.
  std::fstream FS(CommsFile, std::ios_base::in | std::ios_base::binary);
  size_t HandleSize = 0;
  FS.read(reinterpret_cast<char *>(&HandleSize), sizeof(size_t));
  std::unique_ptr<std::byte[]> HandleData{new std::byte[HandleSize]};
  FS.read(reinterpret_cast<char *>(HandleData.get()), HandleSize);

  // Open IPC handle.
#ifdef USE_VIEW
  syclexp::ipc_memory::handle_data_view_t Handle{HandleData.get(), HandleSize};
#else
  syclexp::ipc_memory::handle_data_t Handle{HandleData.get(),
                                            HandleData.get() + HandleSize};
#endif
  int *DataPtr = reinterpret_cast<int *>(
      syclexp::ipc_memory::open(Handle, Q.get_context(), Q.get_device()));

  // Test the data already in the USM pointer.
  int Failures = 0;
  int Read[N] = {0};
  Q.copy(DataPtr, Read, N).wait();
  for (size_t I = 0; I < N; ++I) {
    if (Read[I] != I) {
      ++Failures;
      std::cout << "Failed from consumer: Result at " << I
                << " unexpected: " << Read[I] << " != " << I << std::endl;
    }
  }

  Q.parallel_for(N, [=](sycl::item<1> I) {
     DataPtr[I] = static_cast<int>(N - I.get_linear_id());
   }).wait();

  // Close the IPC pointer.
  syclexp::ipc_memory::close(DataPtr, Q.get_context());

  return Failures;
}

int main(int argc, char *argv[]) {
  return argc == 1 ? spawner(argc, argv) : consumer();
}
