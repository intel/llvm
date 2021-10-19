// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %RUN_ON_HOST %t.out

// This test performs basic check of the SYCL context class.

#include <sycl/sycl.hpp>
#include <iostream>

using namespace cl::sycl;

int main() {
  try {
    context c;
  } catch (device_error e) {
    std::cout << "Failed to create device for context" << std::endl;
  }

  auto devices = device::get_devices();
  device &deviceA = devices[0];
  device &deviceB = (devices.size() > 1 ? devices[1] : devices[0]);
  {
    std::cout << "move constructor" << std::endl;
    context Context(deviceA);
    size_t hash = std::hash<context>()(Context);
    context MovedContext(std::move(Context));
    assert(hash == std::hash<context>()(MovedContext));
    assert(deviceA.is_host() == MovedContext.is_host());
  }
  {
    std::cout << "move assignment operator" << std::endl;
    context Context(deviceA);
    size_t hash = std::hash<context>()(Context);
    context WillMovedContext(deviceB);
    WillMovedContext = std::move(Context);
    assert(hash == std::hash<context>()(WillMovedContext));
    assert(deviceA.is_host() == WillMovedContext.is_host());
  }
  {
    std::cout << "copy constructor" << std::endl;
    context Context(deviceA);
    size_t hash = std::hash<context>()(Context);
    context ContextCopy(Context);
    assert(hash == std::hash<context>()(Context));
    assert(hash == std::hash<context>()(ContextCopy));
    assert(Context == ContextCopy);
    assert(Context.is_host() == ContextCopy.is_host());
  }
  {
    std::cout << "copy assignment operator" << std::endl;
    context Context(deviceA);
    size_t hash = std::hash<context>()(Context);
    context WillContextCopy(deviceB);
    WillContextCopy = Context;
    assert(hash == std::hash<context>()(Context));
    assert(hash == std::hash<context>()(WillContextCopy));
    assert(Context == WillContextCopy);
    assert(Context.is_host() == WillContextCopy.is_host());
  }
  {
    auto AsyncHandler = [](const sycl::exception_list &EL) {};
    sycl::context Context1(sycl::property_list{});
    sycl::context Context2(AsyncHandler, sycl::property_list{});
    sycl::context Context3(deviceA, sycl::property_list{});
    sycl::context Context4(deviceA, AsyncHandler, sycl::property_list{});
    sycl::context Context5(deviceA.get_platform(), sycl::property_list{});
    sycl::context Context6(deviceA.get_platform(), AsyncHandler,
                           sycl::property_list{});
    sycl::context Context7(std::vector<sycl::device>{deviceA},
                           sycl::property_list{});
    sycl::context Context8(
        std::vector<sycl::device>{deviceA}, AsyncHandler,
        sycl::property_list{
            sycl::property::context::cuda::use_primary_context{}});

    if (!Context8.has_property<
            sycl::property::context::cuda::use_primary_context>()) {
      std::cerr << "Line " << __LINE__ << ": Property was not found"
                << std::endl;
      return 1;
    }

    sycl::property::context::cuda::use_primary_context Prop =
        Context8
            .get_property<sycl::property::context::cuda::use_primary_context>();
  }
}
