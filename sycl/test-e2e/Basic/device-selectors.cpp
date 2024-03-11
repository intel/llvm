// RUN: %{build} -o %t.out -DSYCL_DISABLE_IMAGE_ASPECT_WARNING
// RUN: %{run-unfiltered-devices} %t.out

#include <sycl/sycl.hpp>
using namespace sycl;

auto exception_handler_lambda = [](exception_list elist) {
  // nop
};

void handle_exceptions_f(exception_list elist) {
  // nop
}

auto refuse_any_device_lambda = [](const device &d) { return -1; };

int refuse_any_device_f(const device &d) { return -1; }

int main() {
  // Ensure these compile correctly. That exception handler does not
  // end up specialized against device selector.
  queue exception_queue_01(exception_handler_lambda);
  queue exception_queue_02(handle_exceptions_f);

  // Set up callable device selector.
  std::vector<device> deviceList = device::get_devices();
  device &lastDevice = deviceList.back();
  auto select_last_device = [&lastDevice](const device &d) {
    return d == lastDevice;
  };

  // Instantiate platform via callable Device Selector.
  platform lastPlatform(select_last_device);

  // Test each of the four queue constructors that take callable Device
  // Selectors: q(device selector) q(dev selector ,  async handler)
  // q(context , dev selector) q(context, dev selector , async handler)
  queue lastQueue(select_last_device);
  assert(lastQueue.get_device().get_platform() == lastPlatform &&
         "Queue and platform created by selecting same device, should result "
         "in matching platforms.");

  queue lastQueueWithHandler(select_last_device, exception_handler_lambda);
  assert(lastQueueWithHandler.get_device() == lastDevice &&
         "Queue created by selecting a device should have that same device.");

  // Create a context.
  platform plt;
  std::vector<device> platformDevices = plt.get_devices();
  context ctx(platformDevices);
  // Set up a callable device selector to select the last device from the
  // context.
  device &lastPlatformDevice = platformDevices.back();
  auto select_last_platform_device = [&lastPlatformDevice](const device &d) {
    return d == lastPlatformDevice;
  };
  // Test queue constructors that use devices from a context.
  queue lastQueueViaContext(ctx, select_last_platform_device);
  assert(lastQueueViaContext.get_device() == lastPlatformDevice &&
         "Queue created by selecting a device should have that same device.");

  queue lastQueueViaContextWithHandler(ctx, select_last_platform_device,
                                       handle_exceptions_f);
  assert(lastQueueViaContextWithHandler.get_device() == lastPlatformDevice &&
         "Queue created by selecting a device should have that same device.");

  // Device constructor.
  device matchingDevice(select_last_device);
  assert(matchingDevice == lastDevice && "Incorrect selected device.");

  // Check exceptions and failures.
  try {
    platform refusedPlatform(refuse_any_device_lambda);
    assert(false &&
           "Declined device selection should have resulted in an exception.");
  } catch (sycl::exception &e) {
    assert(e.code() == sycl::errc::runtime && "Incorrect error code.");
  }

  try {
    queue refusedQueue(refuse_any_device_f);
    assert(false &&
           "Declined device selection should have resulted in an exception.");
  } catch (sycl::exception &e) {
    assert(e.code() == sycl::errc::runtime && "Incorrect error code.");
  }

  try {
    device refusedDevice(refuse_any_device_f);
    assert(false &&
           "Declined device selection should have resulted in an exception.");
  } catch (sycl::exception &e) {
    assert(e.code() == sycl::errc::runtime && "Incorrect error code.");
  }

  // Standalone device selectors.
  queue default_queue(default_selector_v); // Compilation and no error

  // Not sure what devices are available when this test is run, so no action is
  // necessary on error.
  try {
    queue gpu_queue(gpu_selector_v);
    assert(gpu_queue.get_device().is_gpu() &&
           "Incorrect device. Expected GPU.");
  } catch (exception &e) {
  }
  try {
    queue cpu_queue(cpu_selector_v);
    assert(cpu_queue.get_device().is_cpu() &&
           "Incorrect device. Expected CPU.");
  } catch (exception &e) {
  }
  try {
    queue acc_queue(accelerator_selector_v);
    assert(acc_queue.get_device().is_accelerator() &&
           "Incorrect device. Expected Accelerator.");
  } catch (exception &e) {
  }

  // ------ aspect_selectors --------

  // Unrestrained selection, equivalent to default_selector.
  device dev0 = device{aspect_selector()};
  assert(dev0 == default_queue.get_device() &&
         "unrestrained aspect selector should result in same device as "
         "default_selector");

  // -- Individual test runs may or may not have desired device available.
  // Ignore any exception.

  // Pass aspects in a vector.
  try {
    device dev1 =
        device{aspect_selector(std::vector{aspect::cpu, aspect::image})};
    assert(dev1.is_cpu() && "Incorrect device, expected CPU.");
    assert(dev1.has(aspect::image) && "Device should support aspect::image.");
  } catch (exception &e) {
  }

  // Pass aspects without a vector.
  try {
    device dev2 = device{aspect_selector(aspect::gpu, aspect::image)};
    assert(dev2.is_gpu() && "Incorrect device, expected GPU.");
    assert(dev2.has(aspect::image) && "Device should support aspect::image.");
  } catch (exception &e) {
  }

  // Pass aspects as compile-time parameters
  try {
    device dev3 = device{aspect_selector<aspect::cpu, aspect::fp64>()};
    assert(dev3.is_cpu() && "Incorrect device, expected CPU.");
    assert(dev3.has(aspect::fp64) && "Device should support aspect::fp64.");
  } catch (exception &e) {
  }

  // Pass aspects in an allowlist and a denylist.
  try {
    device dev4 = device{aspect_selector(std::vector{aspect::accelerator},
                                         std::vector{aspect::image})};
    assert(dev4.is_accelerator() && "Incorrect device, expected accelerator.");
    assert(!dev4.has(aspect::image) &&
           "Device should NOT support aspect::image.");
  } catch (exception &e) {
  }

  // Validate errors.
  try {
    device dev5 = device{aspect_selector<aspect::cpu, aspect::gpu>()};
    assert(
        "We should not be here. No known device that is both a CPU and a GPU.");
  } catch (exception &e) {
    assert(e.code() == sycl::errc::runtime && "Incorrect error code.");
  }

  try {
    device dev6 = device{
        aspect_selector(std::vector{aspect::cpu}, std::vector{aspect::cpu})};
    assert("We should not be here. RequireList and DenyList impossible to "
           "fulfill.");
  } catch (exception &e) {
    assert(e.code() == sycl::errc::runtime && "Incorrect error code.");
  }

  return 0;
}
