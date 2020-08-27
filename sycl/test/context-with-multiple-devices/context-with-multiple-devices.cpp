// REQUIRES: cpu, accelerator, aoc

// UNSUPPORTED: cuda, level_zero

// RUN: %clangxx -fsycl -fsycl-unnamed-lambda -fsycl-targets=%sycl_triple %s -o %t1.out
// RUN: %CPU_RUN_PLACEHOLDER CL_CONFIG_CPU_EMULATE_DEVICES=2 %t1.out
// RUN: %CPU_RUN_PLACEHOLDER CL_CONFIG_CPU_EMULATE_DEVICES=4 %t1.out
// RUN: %clangxx -fsycl -fintelfpga -fsycl-unnamed-lambda %s -o %t2.out
// RUN: %ACC_RUN_PLACEHOLDER CL_CONFIG_CPU_EMULATE_DEVICES=2 %t2.out

#include <CL/sycl.hpp>

void exceptionHandler(sycl::exception_list exceptions) {
  for (std::exception_ptr const &e : exceptions) {
    try {
      std::rethrow_exception(e);
    } catch (sycl::exception const &e) {
      std::cerr << "Caught asynchronous SYCL exception:\n"
                << e.what() << std::endl;
    }
  }
}

int main() {
  auto DeviceList = sycl::device::get_devices();

  // remove host device from the list
  DeviceList.erase(std::remove_if(DeviceList.begin(), DeviceList.end(),
                                  [](auto Device) { return Device.is_host(); }),
                   DeviceList.end());

  sycl::context Context(DeviceList, &exceptionHandler);

  std::vector<sycl::queue> QueueList;
  for (const auto &Device : Context.get_devices()) {
    QueueList.emplace_back(Context, Device, &exceptionHandler);
  }

  for (auto &Queue : QueueList) {
    Queue.submit(
        [&](sycl::handler &cgh) { cgh.parallel_for(100, [=](auto i) {}); });
  }

  return 0;
}
