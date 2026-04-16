// REQUIRES: aspect-ext_oneapi_exportable_device_mem
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// clang-format off
/*
This test verifies the basic functionality of memory export APIs in SYCL. 
It allocates exportable device memory, exports the handle, 
and performs a simple validation on the exported handle  to ensure it is valid.
*/
// clang-format on

#include <sycl/ext/oneapi/memory_export.hpp>

#include <iostream>

#ifdef _WIN32
#include <Windows.h>
#else
#include <fcntl.h>
#endif

namespace syclexp = sycl::ext::oneapi::experimental;

int main() {

  void *pExportableLinearMemory = nullptr;
  size_t bufferSize = 1024;
  size_t alignment = 0;
  const size_t memorySizeBytes = bufferSize * sizeof(int);
  sycl::queue syclQueue;
  try {

#ifdef _WIN32
    void *exportableMemHandle = nullptr;
    pExportableLinearMemory = syclexp::alloc_exportable_device_mem(
        alignment, memorySizeBytes,
        syclexp::external_mem_handle_type::win32_nt_handle, syclQueue);

    exportableMemHandle = syclexp::export_device_mem_handle<
        syclexp::external_mem_handle_type::win32_nt_handle>(
        pExportableLinearMemory, syclQueue);

    HANDLE winHandle = static_cast<HANDLE>(exportableMemHandle);

    [[maybe_unused]] DWORD winFlags;
    if (GetHandleInformation(winHandle, &winFlags) == 0) {
      std::cerr << "Failed to validate win32_nt_handle" << std::endl;
      syclexp::free_exportable_memory(pExportableLinearMemory, syclQueue);
      return 1;
    }
#else
    int exportableMemHandle = -1;
    pExportableLinearMemory = syclexp::alloc_exportable_device_mem(
        alignment, memorySizeBytes,
        syclexp::external_mem_handle_type::opaque_fd, syclQueue);

    exportableMemHandle = syclexp::export_device_mem_handle<
        syclexp::external_mem_handle_type::opaque_fd>(pExportableLinearMemory,
                                                      syclQueue);
    if (fcntl(exportableMemHandle, F_GETFL) < 0) {
      std::cerr << "Failed to validate opaque_fd" << std::endl;
      syclexp::free_exportable_memory(pExportableLinearMemory, syclQueue);
      return 2;
    }
#endif

  } catch (const sycl::exception &e) {
    std::cerr << "SYCL exception: " << e.what() << std::endl;
    syclexp::free_exportable_memory(pExportableLinearMemory, syclQueue);
    return 3;
  }
  syclexp::free_exportable_memory(pExportableLinearMemory, syclQueue);
  return 0;
}