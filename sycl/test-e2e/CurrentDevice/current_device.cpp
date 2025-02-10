
#include <sycl/ext/oneapi/experimental/current_device.hpp>
#include <sycl/sycl.hpp>

#include <iostream>
#include <thread>

void thread_func(sycl::device dev) {
    try {
        auto device = sycl::ext::oneapi::experimental::this_thread::get_current_device();
        std::cout << "Thread ID: " << std::this_thread::get_id()
                  << " | Device: " << device.get_info<sycl::info::device::name>() << std::endl;
    } catch (sycl::exception const &e) {
        std::cerr << "SYCL Exception: " << e.what() << std::endl;
    }
}

int main() {
    std::thread t1(thread_func, sycl::device{sycl::default_selector_v});
    std::thread t2(thread_func, sycl::device{sycl::gpu_selector_v});

    t1.join();
    t2.join();

    return 0;
}