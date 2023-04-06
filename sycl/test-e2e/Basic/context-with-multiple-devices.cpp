// REQUIRES: accelerator, opencl-aot

// RUN: %clangxx -fsycl -fintelfpga -fsycl-unnamed-lambda %s -o %t2.out
// RUN: env CL_CONFIG_CPU_EMULATE_DEVICES=2 %t2.out

#include <sycl/sycl.hpp>

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
  auto DeviceList =
      sycl::device::get_devices(sycl::info::device_type::accelerator);

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
