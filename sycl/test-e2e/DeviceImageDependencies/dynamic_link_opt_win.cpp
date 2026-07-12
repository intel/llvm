// Test that device image dependencies from a DLL are not dropped when the
// application is built at higher optimization levels on Windows.
//
// The bug: at /O2 the linker sees that no host symbol from the DLL is
// referenced and drops the DLL import, which prevents the SYCL runtime
// from discovering the device image in the DLL.

// REQUIRES: windows

// Build the DLL using the existing input:
// RUN: %clangxx -fsycl -fsycl-allow-device-image-dependencies -DMAKE_DLL -O2 %shared_lib %S/Inputs/d.cpp -o %t_d.dll

// Build app at -O0 — should work:
// RUN: %clangxx -fsycl -fsycl-allow-device-image-dependencies %O0 %s %t_d.lib -o %t_O0.exe
// RUN: %{run} %t_O0.exe

// Build app at -O2 — fails due to the bug (DLL import dropped):
// RUN: %clangxx -fsycl -fsycl-allow-device-image-dependencies -O2 %s %t_d.lib -o %t_O2.exe
// RUN: %{run} %t_O2.exe

// XFAIL: windows && run-mode
// XFAIL-TRACKER: CMPLRLLVM-76054

#include <cassert>
#include <sycl/sycl.hpp>

// No __declspec(dllimport) — only a device-side dependency on the DLL.
SYCL_EXTERNAL int levelD(int val);

int main() {
  int result = 0;
  {
    sycl::queue q;
    int *d = sycl::malloc_device<int>(1, q);
    q.single_task([=] { *d = levelD(0); }).wait();
    q.memcpy(&result, d, sizeof(int)).wait();
    sycl::free(d, q);
  }
  // Device path of levelD: val |= (0xD << 12) => 0xD000
  assert(result == 0xD000);
  return 0;
}
