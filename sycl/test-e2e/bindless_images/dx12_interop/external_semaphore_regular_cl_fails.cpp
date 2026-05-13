
//
// REQUIRES: aspect-ext_oneapi_external_semaphore_import, windows
//
// RUN: %{build} -o %t.exe
// RUN: %{run} %t.exe

// Importing a DX12 fence external semaphore on a queue backed by a regular
// (non-immediate) command list must throw sycl::exception.
//
// Mirrors vulkan_interop/external_semaphore_regular_cl_fails.cpp but uses
// the win32_nt_dx12_fence handle-type path. Today the runtime rejects at
// import independent of handle type; this is belt-and-suspenders coverage
// in case a future adapter change treats handle types differently.
//
// This is a contract test, not an interop test: no real D3D12 device is
// created. The runtime rejects at import before inspecting the handle, so
// a null handle is enough. That's also why this test does not link against
// DirectX -- it lives here because the DX12 fence handle type is what
// readers will look for in this directory.

#include <iostream>
#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/bindless_images.hpp>
#include <sycl/properties/all_properties.hpp>

namespace syclexp = sycl::ext::oneapi::experimental;

int main() {
  sycl::queue q{
      {sycl::property::queue::in_order{},
       sycl::ext::intel::property::queue::no_immediate_command_list{}}};

  syclexp::external_semaphore_descriptor<syclexp::resource_win32_handle> desc{
      /*handle=*/nullptr,
      syclexp::external_semaphore_handle_type::win32_nt_dx12_fence};

  try {
    (void)syclexp::import_external_semaphore(desc, q);
  } catch (const sycl::exception &e) {
    std::cout << "Got expected sycl::exception: " << e.what() << std::endl;
    return 0;
  }

  std::cerr << "FAIL: import_external_semaphore (dx12_fence) on a "
               "non-immediate-CL queue did not throw."
            << std::endl;
  return 1;
}
