#pragma once

#include <cassert>
#include <fstream>
#include <iostream>
#include <level_zero/ze_api.h>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <variant>
#include <vector>

#define ASSERT_ZE_RESULT_SUCCESS(status) assert((status) == ZE_RESULT_SUCCESS);

inline ze_result_t getDriver(ze_driver_handle_t &ZeDriver) {
  uint32_t DriverCount = 0;
  ze_result_t status = zeDriverGet(&DriverCount, nullptr);
  if (status != ZE_RESULT_SUCCESS) {
    return status;
  }

  if (DriverCount == 0) {
    std::cout << "No Level Zero drivers found" << std::endl;
    return ZE_RESULT_ERROR_DEVICE_LOST;
  }

  std::vector<ze_driver_handle_t> Drivers(DriverCount);
  status = zeDriverGet(&DriverCount, Drivers.data());
  ZeDriver = Drivers[0];
  return status;
}

inline std::vector<uint8_t> loadSpirvFromFile(const std::string &FileName) {
  std::ifstream SpvStream(FileName, std::ios::binary);
  SpvStream.seekg(0, std::ios::end);
  size_t sz = SpvStream.tellg();
  SpvStream.seekg(0);
  std::vector<uint8_t> Spv(sz);
  SpvStream.read(reinterpret_cast<char *>(Spv.data()), sz);
  return Spv;
}

inline bool getCommandListFromQueue(sycl::queue &Queue,
                                    ze_command_list_handle_t &ZeCommandList) {
  using namespace sycl;
  auto ZeQueueNative = get_native<backend::ext_oneapi_level_zero>(Queue);

  if (!std::holds_alternative<ze_command_list_handle_t>(ZeQueueNative)) {
    return false;
  }

  ZeCommandList = std::get<ze_command_list_handle_t>(ZeQueueNative);
  return true;
}

typedef ze_result_t(ZE_APICALL *zeCommandListAppendHostFunction_fn)(
    ze_command_list_handle_t, void *, void *, void *, ze_event_handle_t,
    uint32_t, ze_event_handle_t *);

template <typename FunctionPtr>
inline ze_result_t loadZeExtensionFunction(ze_driver_handle_t ZeDriver,
                                           const char *FunctionName,
                                           FunctionPtr &Fn) {
  ze_result_t status = zeDriverGetExtensionFunctionAddress(
      ZeDriver, FunctionName, reinterpret_cast<void **>(&Fn));
  return status;
}

// Factory for creating and managing Level Zero kernels and modules
// All resources are associated with the same context and device
// The factory stores a copy of the SYCL queue to ensure the underlying
// Level Zero context and device remain valid for the factory's lifetime
class ZeKernelFactory {
public:
  explicit ZeKernelFactory(sycl::queue Queue)
      : Queue(Queue),
        Context(sycl::get_native<sycl::backend::ext_oneapi_level_zero>(
            Queue.get_context())),
        Device(sycl::get_native<sycl::backend::ext_oneapi_level_zero>(
            Queue.get_device())) {}

  ~ZeKernelFactory() { cleanup(); }

  ZeKernelFactory() = delete;
  ZeKernelFactory(const ZeKernelFactory &) = delete;
  ZeKernelFactory &operator=(const ZeKernelFactory &) = delete;
  ZeKernelFactory(ZeKernelFactory &&) noexcept = default;
  ZeKernelFactory &operator=(ZeKernelFactory &&) noexcept = default;

  ze_module_handle_t createModule(const std::vector<uint8_t> &Spirv) {
    ze_module_desc_t moduleDesc = {ZE_STRUCTURE_TYPE_MODULE_DESC,
                                   nullptr,
                                   ZE_MODULE_FORMAT_IL_SPIRV,
                                   Spirv.size(),
                                   Spirv.data(),
                                   nullptr,
                                   nullptr};
    ze_module_handle_t module;
    ze_result_t status =
        zeModuleCreate(Context, Device, &moduleDesc, &module, nullptr);
    ASSERT_ZE_RESULT_SUCCESS(status);
    Modules.push_back(module);
    return module;
  }

  ze_kernel_handle_t createKernel(ze_module_handle_t Module,
                                  const char *KernelName) {
    ze_kernel_desc_t kernelDesc = {ZE_STRUCTURE_TYPE_KERNEL_DESC, nullptr, 0,
                                   KernelName};
    ze_kernel_handle_t kernel;
    ze_result_t status = zeKernelCreate(Module, &kernelDesc, &kernel);
    ASSERT_ZE_RESULT_SUCCESS(status);
    Kernels.push_back(kernel);
    return kernel;
  }

  void cleanup() {
    // Destroy kernels first (they depend on modules)
    for (auto kernel : Kernels) {
      ASSERT_ZE_RESULT_SUCCESS(zeKernelDestroy(kernel));
    }
    Kernels.clear();

    // Then destroy modules
    for (auto module : Modules) {
      ASSERT_ZE_RESULT_SUCCESS(zeModuleDestroy(module));
    }
    Modules.clear();
  }

private:
  sycl::queue Queue; // Ensures context and device lifetime
  ze_context_handle_t Context;
  ze_device_handle_t Device;
  std::vector<ze_module_handle_t> Modules;
  std::vector<ze_kernel_handle_t> Kernels;
};
