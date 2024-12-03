// REQUIRES: linux, cpu || (gpu && level_zero)
// RUN: %{build} %device_asan_flags -O0 -g -o %t
// RUN: %{run} %t

// clang-format off
// Invalid ur option format
// RUN: env UR_LAYER_ASAN_OPTIONS=a:1,b:1 %{run} not %t 2>&1 | FileCheck %s  --check-prefixes INVALID-FORMAT
// INVALID-FORMAT: <SANITIZER>[ERROR]: Wrong format of the UR_LAYER_ASAN_OPTIONS environment variable value

// Invalid bool option
// RUN: env UR_LAYER_ASAN_OPTIONS=debug:yes %{run} not %t 2>&1 | FileCheck %s  --check-prefixes INVALID-BOOL
// INVALID-BOOL: <SANITIZER>[ERROR]: "debug" is set to "yes", which is not an valid setting. Acceptable input are: for enable, use: "1" "true"; for disable, use: "0" "false".

// Invalid quarantine_size_mb
// RUN: env UR_LAYER_ASAN_OPTIONS=quarantine_size_mb:-1 %{run} not %t 2>&1 | FileCheck %s  --check-prefixes INVALID-QUARANTINE
// RUN: env UR_LAYER_ASAN_OPTIONS=quarantine_size_mb:4294967296 %{run} not %t 2>&1 | FileCheck %s  --check-prefixes INVALID-QUARANTINE
// INVALID-QUARANTINE: <SANITIZER>[ERROR]: "quarantine_size_mb" should be an integer in range[0, 4294967295].

// Invalid redzone and max_redzone
// RUN: env UR_LAYER_ASAN_OPTIONS=redzone:abc %{run} not %t 2>&1 | FileCheck %s  --check-prefixes INVALID-REDZONE
// INVALID-REDZONE: <SANITIZER>[ERROR]: "redzone" should be an integer in range[0, 16].
// RUN: env UR_LAYER_ASAN_OPTIONS=max_redzone:abc %{run} not %t 2>&1 | FileCheck %s  --check-prefixes INVALID-MAXREDZONE
// INVALID-MAXREDZONE: <SANITIZER>[ERROR]: "max_redzone" should be an integer in range[0, 2048].
// clang-format on

#include <sycl/usm.hpp>

int main() {
  sycl::queue q;
  constexpr std::size_t N = 8;
  auto *array = sycl::malloc_device<char>(N, q);

  q.submit([&](sycl::handler &h) {
     h.single_task<class Test>([=]() { ++array[0]; });
   }).wait();

  sycl::free(array, q);
  return 0;
}
