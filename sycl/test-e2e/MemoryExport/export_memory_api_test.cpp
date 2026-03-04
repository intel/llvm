// REQUIRES: aspect-ext_oneapi_exportable_device_mem
// REQUIRES: target-spir

// RUN: %{build} -o %t.out %if target-spir %{ -Wno-ignored-attributes %}
// RUN: %{run} %t.out

#include <iostream>
#include <sycl/ext/oneapi/memory_export.hpp>

namespace syclexp = sycl::ext::oneapi::experimental;

#ifdef _WIN32
using exported_handle_type = void *;
#else
using exported_handle_type = int;
#endif // _WIN32

int main() {
  sycl::device Device;
  sycl::context Context = sycl::context(Device);
  sycl::queue Queue(Context, Device);

  // Check if the device supports memory export.
  bool HasExportSupport =
      Device.has(sycl::aspect::ext_oneapi_exportable_device_mem);

  if (!HasExportSupport) {
    std::cerr << "Device does not support memory export.\n";
    return 1;
  } else {
    std::cout << "Device supports memory export.\n";
  }

#ifdef _WIN32
  constexpr auto ExportHandleType =
      syclexp::external_mem_handle_type::win32_nt_handle;
#else
  constexpr auto ExportHandleType =
      syclexp::external_mem_handle_type::opaque_fd;
#endif // _WIN32

  try {
    // Allocate exportable memory.
    size_t Size = 1024;
    void *Mem = syclexp::alloc_exportable_device_mem(
        0 /* alignment */, Size, ExportHandleType, Device, Context);

    // Export the memory handle.
    exported_handle_type ExportableMemoryHandle =
        syclexp::export_device_mem_handle<ExportHandleType>(Mem, Device,
                                                            Context);
    std::cout << "Exported memory handle == " << ExportableMemoryHandle << "\n";

    // Free the exportable memory.
    syclexp::free_exportable_memory(Mem, Device, Context);

  } catch (const sycl::exception &e) {
    std::cerr << "SYCL exception caught: " << e.what() << "\n";
    return 2;
  } catch (...) {
    std::cerr << "Unknown exception caught.\n";
    return 3;
  }

  return 0;
}
