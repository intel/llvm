// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// RUN: %if preview-breaking-changes-supported %{ %{build} -fpreview-breaking-changes -D_GLIBCXX_USE_CXX11_ABI=0 -o %t2.out %}
// RUN: %if preview-breaking-changes-supported %{ %{run} %t2.out %}

// This test case tests if compiling works with or without
// _GLIBCXX_USE_CXX11_ABI=0.

#include <iostream>
#include <sycl/sycl.hpp>

int main() {
  try {
    auto cgf = [&](sycl::handler &cgh) { cgh.single_task([=]() {}); };
    sycl::queue queue;
    for (auto i = 0; i < 25; i++) {
      sycl::malloc_device(1024, queue);
    }
    auto event = queue.submit(cgf);
    event.wait_and_throw();
  } catch (const sycl::exception &ep) {
    const std::string_view err_msg(ep.what());
    if (err_msg.find("PI_ERROR_OUT_OF_RESOURCES") != std::string::npos) {
      std::cout << "Allocation is out of device memory on the current platform."
                << std::endl;
    } else {
      throw ep;
    }
  }
  std::cout << "pass!" << std::endl;
}
