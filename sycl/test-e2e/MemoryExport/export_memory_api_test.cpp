// REQUIRES: aspect-ext_oneapi_memory_export_linear
// REQUIRES: target-spir

// RUN: %{build} -o %t.out %if target-spir %{ -Wno-ignored-attributes %}
// RUN: %{run} %t.out

#include <iostream>
#include <sycl/ext/oneapi/memory_export.hpp>
#include <sycl/sycl.hpp>

namespace syclexp = sycl::ext::oneapi::experimental;

int main() {
  sycl::device device;
  sycl::context context = sycl::context(device);
  sycl::queue queue(context, device);

  // Check if the device supports memory export.
  bool hasExportSupport =
      device.has(sycl::aspect::ext_oneapi_memory_export_linear);

  if (!hasExportSupport) {
    std::cerr << "Device does not support memory export.\n";
    return 1;
  } else {
    std::cout << "Device supports memory export.\n";
  }

  // Allocate exportable memory.
  size_t size = 1024;

#ifndef _WIN32
  void *mem = syclexp::alloc_exportable_device_mem(
      0, size, syclexp::export_external_mem_handle_type::opaque_fd, device,
      context);
#else
  void *mem = syclexp::alloc_exportable_device_mem(
      0, size, syclexp::export_external_mem_handle_type::win32_nt,
      device, context);
#endif // _WIN32

  // Export the memory handle.
#ifndef _WIN32
  auto exportableMemoryHandle = syclexp::export_device_mem_handle<
      syclexp::export_external_mem_handle_type::opaque_fd>(mem, device,
                                                           context);
  std::cout << "Exported file descriptor == " << exportableMemoryHandle << "\n";
#else
  auto exportableMemoryHandle = syclexp::export_device_mem_handle<
      syclexp::export_external_mem_handle_type::win32_nt>(mem, device,
                                                                 context);
  std::cout << "Exported win32 handle == " << exportableMemoryHandle << "\n";
#endif // _WIN32

  // Free the exportable memory.
  syclexp::free_exportable_memory(mem, device, context);

  return 0;
}
