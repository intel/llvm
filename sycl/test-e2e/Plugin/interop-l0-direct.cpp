// REQUIRES: level_zero, level_zero_dev_kit
// UNSUPPORTED: ze_debug
// RUN: %{build} %level_zero_options -o %t.out
// RUN: env SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0 %{run} %t.out
// RUN: env SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1 %{run} %t.out

#include <iostream>
#include <level_zero/ze_api.h>
#include <sycl/backend.hpp>
#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <sycl/usm.hpp>
#include <variant>
#include <vector>

// Compile as clang++ -fsycl %s -lze_loader
using namespace sycl;

int main() {

  // L0 object creation
  // Init L0
  ze_result_t result = zeInit(0);
  if (result != ZE_RESULT_SUCCESS) {
    std::cout << "zeInit failed\n";
    return 1;
  }

  // Create Driver
  uint32_t driver_handle_count = 0;
  result = zeDriverGet(&driver_handle_count, nullptr);
  if (result != ZE_RESULT_SUCCESS) {
    std::cout << "zeDriverGet failed\n";
    return 1;
  }
  std::cout << "Found " << driver_handle_count << " driver(s)\n";
  if (driver_handle_count == 0)
    return 1;

  std::vector<ze_driver_handle_t> driver_handles(driver_handle_count);
  result = zeDriverGet(&driver_handle_count, driver_handles.data());
  if (result != ZE_RESULT_SUCCESS) {
    std::cout << "zeDriverGet failed\n";
    return 1;
  }

  ze_driver_handle_t ZeDriver = driver_handles[0];
  std::cout << "Using default driver, index 0\n";

  // Create Context
  ze_context_handle_t ZeContext;
  ze_context_desc_t ctxtDesc = {ZE_STRUCTURE_TYPE_CONTEXT_DESC, nullptr, 0};
  if (zeContextCreate(ZeDriver, &ctxtDesc, &ZeContext) != ZE_RESULT_SUCCESS) {
    std::cout << "Context create failed\n";
    return 1;
  }

  // Create Devices
  uint32_t device_count = 0;
  result = zeDeviceGet(ZeDriver, &device_count, nullptr);
  if (result != ZE_RESULT_SUCCESS) {
    std::cout << "zeDeviceGet failed to get count of devices\n";
    return 1;
  }

  std::vector<ze_device_handle_t> ZeDevices(device_count);
  result = zeDeviceGet(ZeDriver, &device_count, ZeDevices.data());
  if (result != ZE_RESULT_SUCCESS) {
    std::cout << "zeDeviceGet failed to get device handles\n";
    return 1;
  }
  std::cout << "Using default device, index 0\n";

  // Create Command Queue
  ze_command_queue_desc_t Qdescriptor = {};
  Qdescriptor.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC;
  Qdescriptor.mode = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS;
  Qdescriptor.ordinal = 0;
  Qdescriptor.index = 0;

  ze_command_queue_handle_t ZeCommand_queue = nullptr;
  result = zeCommandQueueCreate(ZeContext, ZeDevices[0], &Qdescriptor,
                                &ZeCommand_queue);
  if (result != ZE_RESULT_SUCCESS) {
    std::cout << "zeCommandQueueCreate failed\n";
    return 1;
  }
  std::cout << "Commandqueue created: " << ZeCommand_queue << std::endl;

  // Create Command List
  ze_command_list_handle_t ZeCommand_list = nullptr;
  result = zeCommandListCreateImmediate(ZeContext, ZeDevices[0], &Qdescriptor,
                                        &ZeCommand_list);
  if (result != ZE_RESULT_SUCCESS) {
    std::cout << "zeCommandListCreate failed\n";
    return 1;
  }
  std::cout << "Commandlist created: " << ZeCommand_list << std::endl;

  // Interop object creation
  backend_traits<backend::ext_oneapi_level_zero>::return_type<device> ZeDevice;
  ZeDevice = ZeDevices[0];

  backend_input_t<backend::ext_oneapi_level_zero, platform>
      InteropPlatformInput{ZeDriver};
  platform InteropPlatform =
      make_platform<backend::ext_oneapi_level_zero>(InteropPlatformInput);
  std::cout << "Made platform\n";

  backend_input_t<backend::ext_oneapi_level_zero, device> InteropDeviceInput{
      ZeDevice};
  device InteropDevice =
      make_device<backend::ext_oneapi_level_zero>(InteropDeviceInput);
  std::cout << "Made device\n";

  backend_input_t<backend::ext_oneapi_level_zero, context> InteropContextInput{
      ZeContext, std::vector<device>(1, InteropDevice),
      ext::oneapi::level_zero::ownership::keep};
  context InteropContext =
      make_context<backend::ext_oneapi_level_zero>(InteropContextInput);
  std::cout << "Made context\n";

  backend_input_t<backend::ext_oneapi_level_zero, queue> InteropQueueInputCQ{
      ZeCommand_queue, InteropDevice, ext::oneapi::level_zero::ownership::keep};
  queue InteropQueueCQ = make_queue<backend::ext_oneapi_level_zero>(
      InteropQueueInputCQ, InteropContext);
  std::cout << "Made SYCL queue with L0 command queue\n";

  auto InteropQueueCQ_NewHandle =
      get_native<backend::ext_oneapi_level_zero, queue>(InteropQueueCQ);
  if (std::holds_alternative<ze_command_list_handle_t>(
          InteropQueueCQ_NewHandle)) {
    std::cout << "Test failed, queue created using command queue returns a "
                 "command list type handle"
              << std::endl;
    return 1;
  } else {
    auto Queue =
        std::get_if<ze_command_queue_handle_t>(&InteropQueueCQ_NewHandle);
    std::cout << "Command queue obtained new style: " << *Queue << std::endl;
    if (ZeCommand_queue != *Queue) {
      std::cout << "Test failed, command queue retrieved in new style does not "
                   "match one used in queue creation\n";
      return 1;
    }
  }

  backend_input_t<backend::ext_oneapi_level_zero, queue> InteropQueueInputCL{
      ZeCommand_list, InteropDevice, ext::oneapi::level_zero::ownership::keep};
  queue InteropQueueCL = make_queue<backend::ext_oneapi_level_zero>(
      InteropQueueInputCL, InteropContext);
  std::cout << "Made SYCL queue with L0 immediate command list\n";
  auto InteropQueueCL_NewHandle =
      get_native<backend::ext_oneapi_level_zero, queue>(InteropQueueCL);
  if (std::holds_alternative<ze_command_list_handle_t>(
          InteropQueueCL_NewHandle)) {
    auto List =
        std::get_if<ze_command_list_handle_t>(&InteropQueueCL_NewHandle);
    std::cout << "Command list obtained new style: " << *List << std::endl;
    if (ZeCommand_list != *List) {
      std::cout << "Test failed, command list retrieved in new style does not "
                   "match one used in queue creation\n";
      return 1;
    }
  } else {
    std::cout << "Test failed, queue created using command list returns a "
                 "command queue type handle"
              << std::endl;
    return 1;
  }

  int data[3] = {7, 8, 0};
  buffer<int, 1> bufData{data, 3};
  buffer<int, 1> bufDataCQ{data, 3};
  buffer<int, 1> bufDataCL{data, 3};
  range<1> dataCount{3};

  queue SyclQueue;
  device SyclDevice = SyclQueue.get_device();
  context SyclContext = SyclQueue.get_context();

  auto deviceData = malloc_device<int>(2, InteropDevice, InteropContext);
  int addend[2];

  for (int i = 0; i < 3; i++) {
    addend[0] = i;
    addend[1] = i + 1;

    // Try SYCL queue
    SyclQueue.copy<int>(addend, deviceData, 2).wait();
    SyclQueue.submit([&](handler &cgh) {
      accessor numbers{bufData, cgh, read_write};
      cgh.parallel_for(dataCount,
                       [=](id<1> Id) { numbers[Id] += deviceData[0]; });
    });
    host_accessor hostOut{bufData, read_only};
    std::cout << "GPU Result from SYCL Q = {" << hostOut[0] << ", "
              << hostOut[1] << ", " << hostOut[2] << "}" << std::endl;

    // Try interop queue with standard commandlist
    InteropQueueCQ.copy<int>(addend, deviceData, 2).wait();
    InteropQueueCQ.submit([&](handler &cgh) {
      accessor numbers{bufDataCQ, cgh, read_write};
      cgh.parallel_for(dataCount,
                       [=](id<1> Id) { numbers[Id] += deviceData[0]; });
    });
    host_accessor hostOutCQ{bufDataCQ, read_only};
    std::cout << "GPU Result from Standard Q = {" << hostOut[0] << ", "
              << hostOut[1] << ", " << hostOut[2] << "}" << std::endl;

    // Try interop queue with immediate commandlist
    InteropQueueCQ.copy<int>(addend, deviceData, 2).wait();
    InteropQueueCQ.submit([&](handler &cgh) {
      accessor numbers{bufDataCL, cgh, read_write};
      cgh.single_task(
          [=]() { numbers[2] += numbers[0] + numbers[1] + deviceData[1]; });
    });
    host_accessor hostOutCL{bufDataCL, read_only};
    std::cout << "GPU Result from Immediate Q = {" << hostOut[0] << ", "
              << hostOut[1] << ", " << hostOut[2] << "}" << std::endl;
  }
  // Check results
  buffer<int, 1> bufDataResult{data, 3};
  host_accessor hostResult{bufDataResult, read_only};
  if (hostResult[0] != 13 || hostResult[1] != 14 || hostResult[2] != 73) {
    std::cout << "Test failed, expected final result to be {" << hostResult[0]
              << ", " << hostResult[1] << ", " << hostResult[2] << "}"
              << std::endl;
    return 1;
  }

  return 0;
}
