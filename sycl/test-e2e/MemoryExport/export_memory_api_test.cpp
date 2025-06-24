// REQUIRES: aspect-ext_oneapi_memory_export_linear
// REQUIRES: target-spir

// RUN: %{build} -o %t.out %if target-spir %{ -Wno-ignored-attributes %}
// RUN: %{run} %t.out

#include <iostream>
#include <sycl/ext/oneapi/memory_export.hpp>

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

#ifdef _WIN32
  constexpr auto exportHandleType =
      syclexp::export_external_mem_handle_type::win32_nt;
#else
  constexpr auto exportHandleType =
      syclexp::export_external_mem_handle_type::opaque_fd;
#endif // _WIN32

  try {
    // Allocate exportable memory.
    size_t size = 1024;
    void *mem = syclexp::alloc_exportable_device_mem(0, size, exportHandleType,
                                                     device, context);

    // Export the memory handle.
    syclexp::exported_mem_t<exportHandleType> exportableMemoryHandle =
        syclexp::export_device_mem_handle<exportHandleType>(mem, device,
                                                            context);
    std::cout << "Exported memory handle == " << exportableMemoryHandle << "\n";

    // Free the exportable memory.
    syclexp::free_exportable_memory(mem, device, context);

  } catch (const sycl::exception &e) {
    std::cerr << "SYCL exception caught: " << e.what() << "\n";
    return 2;
  } catch (...) {
    std::cerr << "Unknown exception caught.\n";
    return 3;
  }

  return 0;
}
