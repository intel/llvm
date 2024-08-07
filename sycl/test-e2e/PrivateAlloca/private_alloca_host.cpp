// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// Simple test checking calling private_alloca on the host leads to an exception
// being thrown.

#include <sycl/detail/core.hpp>

#include <sycl/ext/oneapi/experimental/alloca.hpp>
#include <sycl/specialization_id.hpp>

constexpr sycl::specialization_id<int> size(10);

template <typename Func>
static void test(std::string_view expectedMessage, Func f) {
  try {
    f();
  } catch (sycl::exception &e) {
    assert(e.code() == sycl::errc::feature_not_supported &&
           "Unexpected error code");
    assert(std::string_view{e.what()} == expectedMessage &&
           "Unexpected error message");
  }
}

int main() {
  using namespace std::literals;

  std::array<uint8_t, sizeof(sycl::kernel_handler)> h;
  auto &kh = *reinterpret_cast<sycl::kernel_handler *>(h.data());
  test(
      "sycl::ext::oneapi::experimental::aligned_private_alloca is not supported on host"sv,
      [&kh]() {
        sycl::ext::oneapi::experimental::aligned_private_alloca<
            float, alignof(double), size, sycl::access::decorated::no>(kh);
      });
  test(
      "sycl::ext::oneapi::experimental::private_alloca is not supported on host"sv,
      [&kh]() {
        sycl::ext::oneapi::experimental::private_alloca<
            float, size, sycl::access::decorated::no>(kh);
      });
}
