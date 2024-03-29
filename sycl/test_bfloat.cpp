#include <sycl/sycl.hpp>

int main() {
    // Create a SYCL queue
    cl::sycl::queue queue;

    sycl::vec<sycl::ext::oneapi::bfloat16, 1> data{1};

    // Create a buffer from the vector
    cl::sycl::buffer<float, 1> buffer(data.data(), data.size());

    // Submit a command group to the queue
    queue.submit(& {
        // Get access to the buffer
        auto accessor = buffer.get_access<cl::sycl::access::mode::read_write>(cgh);

        // Define the kernel
        cgh.parallel_for<class bfloat16_vector>(cl::sycl::range<1>(data.size()), = {
            accessor[i] += 2.0f;
        });
    });

    // Wait for all operations to complete
    queue.wait_and_throw();

    return 0;
}
