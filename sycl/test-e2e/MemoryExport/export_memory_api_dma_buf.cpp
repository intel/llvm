// REQUIRES: aspect-ext_oneapi_exportable_device_mem
// REQUIRES: target-spir
// REQUIRES: linux

// RUN: %{build} -o %t.out %if target-spir %{ -Wno-ignored-attributes %}
// RUN: %{run} %t.out

/*
    clang++ -fsycl -o emadb.bin export_memory_api_dma_buf.cpp

*/

#include <iostream>
#include <sycl/ext/oneapi/memory_export.hpp>

namespace syclexp = sycl::ext::oneapi::experimental;

// Force the DMA_BUF handle type for this test
constexpr auto ExportHandleType = syclexp::external_mem_handle_type::dma_buf;

int main() {
  int failed_tests = 0;
  const size_t alloc_size = 4096;

  try {
    sycl::device Device;
    sycl::context Context(Device);
    sycl::queue Queue(Context, Device);

    std::cout << "Target Device: "
              << Device.get_info<sycl::info::device::name>() << "\n\n";

    if (!Device.has(sycl::aspect::ext_oneapi_exportable_device_mem)) {
      std::cerr << "FATAL: Device does not support memory export aspect.\n";
      return 1;
    }

    std::cout << "Testing explicit DMA_BUF export" << std::endl;

    void *mem_q = nullptr;
    try {
      // Allocate specifically asking for DMA_BUF support
      mem_q = syclexp::alloc_exportable_device_mem(0, alloc_size,
                                                   ExportHandleType, Queue);

      if (!mem_q) {
        std::cerr << "  FAIL: alloc_exportable_device_mem returned nullptr.\n";
        failed_tests++;
      } else {
        std::cout << "  OK: DMA_BUF compatible memory allocated.\n";

        // Attempt to extract the DMA_BUF file descriptor
        // If the SFINAE block is missing, this line will fail to compile.
        int dma_fd =
            syclexp::export_device_mem_handle<ExportHandleType>(mem_q, Queue);

        if (dma_fd > 2) {
          std::cout << "  OK: Successfully extracted DMA_BUF FD: " << dma_fd
                    << "\n";
        } else {
          std::cerr << "  FAIL: Extracted invalid FD: " << dma_fd << "\n";
          failed_tests++;
        }

        // Verify Compute Integrity
        int *int_ptr = static_cast<int *>(mem_q);
        size_t num_elements = alloc_size / sizeof(int);

        Queue
            .submit([&](sycl::handler &h) {
              h.parallel_for(
                  sycl::range<1>(num_elements),
                  [=](sycl::id<1> idx) { int_ptr[idx] = idx[0] + 77; });
            })
            .wait_and_throw();

        std::vector<int> host_verification(num_elements, 0);
        Queue.memcpy(host_verification.data(), int_ptr, alloc_size)
            .wait_and_throw();

        if (host_verification[0] == 77) {
          std::cout
              << "  OK: Compute Verification: Kernel read/write successful.\n";
        } else {
          std::cerr << "  FAIL: Compute Verification: Data mismatch!\n";
          failed_tests++;
        }

        syclexp::free_exportable_memory(mem_q, Queue);
        std::cout << "  OK: Memory freed.\n";
      }
    } catch (const sycl::exception &e) {
      std::cerr << "  FAIL: API threw SYCL exception: " << e.what() << "\n";
      failed_tests++;
    }

    std::cout << "\n------------------------------------------------\n";
    if (failed_tests == 0) {
      std::cout << "RESULT: DMA_BUF EXPORT TEST PASSED.\n";
      return 0;
    } else {
      std::cerr << "RESULT: " << failed_tests << " TESTS FAILED.\n";
      return 1;
    }

  } catch (const std::exception &e) {
    std::cerr << "\nFATAL RUNTIME ERROR: " << e.what() << "\n";
    return 2;
  }
}