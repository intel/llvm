// REQUIRES: windows, level_zero, level_zero_dev_kit
// REQUIRES: arch-intel_gpu_bmg_g21 || arch-intel_gpu_bmg_g31

// RUN: %{build} %level_zero_options -o %t.out
// RUN: %{run} %t.out

// Prints the version of the Level Zero loader used by the SYCL runtime.
// The test always fails, so that its output is printed in the lit log.

#include <level_zero/loader/ze_loader.h>
#include <level_zero/ze_api.h>
#include <sycl/backend.hpp>
#include <sycl/detail/core.hpp>

#include <iostream>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#endif

int main() {
  sycl::queue Q;
  std::cout << "Device: " << Q.get_device().get_info<sycl::info::device::name>()
            << std::endl;

#ifdef _WIN32
  if (HMODULE Loader = GetModuleHandleA("ze_loader.dll")) {
    char Path[MAX_PATH] = {0};
    if (GetModuleFileNameA(Loader, Path, MAX_PATH))
      std::cout << "L0 loader path: " << Path << std::endl;
  }
#endif

  ze_result_t Res = zeInit(0);
  if (Res != ZE_RESULT_SUCCESS) {
    std::cout << "zeInit failed with error code: " << Res << std::endl;
    return 1;
  }

  size_t NumComponents = 0;
  Res = zelLoaderGetVersions(&NumComponents, nullptr);
  if (Res != ZE_RESULT_SUCCESS) {
    std::cout << "zelLoaderGetVersions failed with error code: " << Res
              << std::endl;
    return 1;
  }

  std::vector<zel_component_version_t> Versions(NumComponents);
  Res = zelLoaderGetVersions(&NumComponents, Versions.data());
  if (Res != ZE_RESULT_SUCCESS) {
    std::cout << "zelLoaderGetVersions failed with error code: " << Res
              << std::endl;
    return 1;
  }

  for (const auto &V : Versions)
    std::cout << "L0 component: " << V.component_name
              << ", version: " << V.component_lib_version.major << "."
              << V.component_lib_version.minor << "."
              << V.component_lib_version.patch
              << ", spec version: " << ZE_MAJOR_VERSION(V.spec_version) << "."
              << ZE_MINOR_VERSION(V.spec_version) << std::endl;

  return 1;
}
