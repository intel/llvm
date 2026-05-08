// REQUIRES: aspect-usm_device_allocations && aspect-ext_oneapi_ipc_memory

// UNSUPPORTED: level_zero && windows
// UNSUPPORTED-TRACKER: UMFW-348

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// RUN: %{build} -DUSE_DEPRECATED_IPC_MEMORY_NAMESPACE -o %t2.out
// RUN: %{run} %t2.out

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/ipc_memory.hpp>
#include <sycl/usm.hpp>

#include <cstdio>
#include <cstdlib>

#if defined(__linux__)
#include <linux/prctl.h>
#include <sys/prctl.h>
#include <unistd.h>
#endif // defined(__linux__)

namespace syclexp = sycl::ext::oneapi::experimental;

#ifdef USE_DEPRECATED_IPC_MEMORY_NAMESPACE
namespace ipc_common = syclexp::ipc_memory;
namespace ipc_memory = syclexp::ipc_memory;
#else
namespace ipc_common = syclexp::ipc;
namespace ipc_memory = syclexp::ipc::memory;
#endif

int main() {
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

  int *DataPtr = sycl::malloc_device<int>(32, Q);
  ipc_common::handle Handle = ipc_memory::get(DataPtr, Q.get_context());

  // Free data before put.
  sycl::free(DataPtr, Q);

  // Try calling put after free.
  ipc_memory::put(Handle, Q.get_context());

  return 0;
}
