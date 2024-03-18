// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// Simple test checking calling private_alloca on the host leads to an exception
// being thrown.

#include <sycl/sycl.hpp>

#include <sycl/ext/oneapi/experimental/alloca.hpp>

constexpr sycl::specialization_id<int> size(10);

int main() {
  try {
    std::array<uint8_t, sizeof(sycl::kernel_handler)> h;
    sycl::ext::oneapi::experimental::private_alloca<
        float, size, sycl::access::decorated::no>(
        *reinterpret_cast<sycl::kernel_handler *>(h.data()));
  } catch (sycl::exception &e) {
    assert(e.code() == sycl::errc::feature_not_supported &&
           "Unexpected error code");
  }
}
