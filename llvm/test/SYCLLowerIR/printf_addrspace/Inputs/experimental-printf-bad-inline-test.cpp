#include <sycl/sycl.hpp>

SYCL_EXTERNAL void custom_wrapper(const char *S) {
  sycl::ext::oneapi::experimental::printf(S);
}
