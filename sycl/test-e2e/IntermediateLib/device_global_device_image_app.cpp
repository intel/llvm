// This test verifies that device_global variables work correctly across
// shared library boundaries when multiple device images are linked together.
// The fix ensures that device globals from all linked images are properly
// registered, not just those from the main image.

// UNSUPPORTED: opencl && gpu
// UNSUPPORTED-TRACKER: coming-soon

// UNSUPPORTED: cuda, hip
// UNSUPPORTED-INTENDED: CUDA and HIP targets support AoT compilation only and cannot do runtime linking. 


// DEFINE: %{fPIC_flag} = %if windows %{%} %else %{-fPIC%}
// DEFINE: %{shared_lib_ext} = %if windows %{dll%} %else %{so%}
// DEFINE: %{cuda_target} = %if target-nvidia %{-fsycl-targets=nvptx64-nvidia-cuda%}
// DEFINE: %{amd_target} = %if target-amd %{-fsycl-targets=amdgcn-amd-amdhsa %amd_arch_options%}
// DEFINE: %{spir_target} = %if !target-nvidia && !target-amd %{-fsycl-targets=spir64%}
// DEFINE: %{lib_export_flags} = -ftarget-export-symbols -fsycl-allow-device-image-dependencies

// RUN: rm -rf %t.dir && mkdir -p %t.dir

// RUN: %{run-aux} %clangxx -fsycl %{cuda_target} %{amd_target} %{spir_target} \
// RUN:   %{fPIC_flag} %{lib_export_flags} %shared_lib \
// RUN:   -Wno-unused-command-line-argument \
// RUN:   -o %t.dir/libdevice_global_test.%{shared_lib_ext} \
// RUN:   %S/Inputs/device_global_device_image_lib.cpp

// RUN: %{run-aux} %clangxx -fsycl %{cuda_target} %{amd_target} %{spir_target} \
// RUN:   %{fPIC_flag} %{lib_export_flags} \
// RUN:   -Wno-unused-command-line-argument \
// RUN:   -o %t.dir/test.exe %s \
// RUN:   %if windows %{ %t.dir/libdevice_global_test.lib%} \
// RUN:   %else %{-L%t.dir -ldevice_global_test -Wl,-rpath=%t.dir%}

// RUN: %{run} %t.dir/test.exe

// UNSUPPORTED: target-native_cpu
// UNSUPPORTED-TRACKER: https://github.com/intel/llvm/issues/20142

// clang-format off
/*
// build the shared library
clang++ -fsycl -fsycl-targets=spir64 -ftarget-export-symbols -fsycl-allow-device-image-dependencies -shared -o libdevice_global_test.dll ./Inputs/device_global_device_image_lib.cpp

// build the app - Lin and Win
clang++ -fsycl -fsycl-targets=spir64 -ftarget-export-symbols -fsycl-allow-device-image-dependencies -o test.bin device_global_device_image_app.cpp -L. -ldevice_global_test -Wl,-rpath=.
clang++ -fsycl -fsycl-targets=spir64 -ftarget-export-symbols -fsycl-allow-device-image-dependencies -o test.exe device_global_device_image_app.cpp libdevice_global_test.lib

// run
./test.bin
*/
// clang-format on

#include <iostream>
#include <sycl/detail/core.hpp>
#include <sycl/usm.hpp>

// Declare external symbols from library
namespace syclex = sycl::ext::oneapi::experimental;
extern __SYCL_EXPORT SYCL_EXTERNAL syclex::device_global<int> test_global;

extern "C" void set_test_global(int val);
extern "C" int get_test_global();
extern "C" int read_global_in_lib();

int main() {
  std::cout << "\n=== Test 1: Copy operations from host ===\n";
  set_test_global(42);
  int val = get_test_global();
  std::cout << "After set_test_global(42): get_test_global() = " << val
            << " (expected 42)\n";

  std::cout << "\n=== Test 2: Read in library's kernel ===\n";
  int lib_read = read_global_in_lib();
  std::cout << "read_global_in_lib() = " << lib_read << " (expected 42)\n";

  std::cout << "\n=== Test 3: Read in main's kernel ===\n";
  sycl::queue q;
  int *dev_result = sycl::malloc_device<int>(1, q);

  q.submit([&](sycl::handler &h) {
     h.single_task([=]() {
       dev_result[0] =
           test_global; // Read in main's kernel - this tests the fix
     });
   }).wait();

  int main_read = 0;
  q.copy(dev_result, &main_read, 1).wait();
  sycl::free(dev_result, q);

  std::cout << "main's kernel read: " << main_read << " (expected 42)\n";

  if (val == 42 && lib_read == 42 && main_read == 42) {
    std::cout << "\n✓ ALL TESTS PASSED\n";
    return 0;
  } else {
    std::cout << "\n✗ TESTS FAILED\n";
    return 1;
  }
}
