// RUN: %{build} -o %t.out

// Make sure that debug/test-only option `--sycl-rtc-in-memory-fs-only` works
// RUN: %{run} not %t.out --sycl-rtc-in-memory-fs-only --sycl-rtc-use-system-includes | FileCheck %s --check-prefix CHECK-ERROR
// CHECK-ERROR-LABEL:  Device compilation failed
// CHECK-ERROR-NEXT:   Detailed information:
// CHECK-ERROR:        	In file included from rtc_0.cpp:2:
// CHECK-ERROR-NEXT:   In file included from {{.*}}/sycl-jit-toolchain//bin/../include/sycl/sycl.hpp:38:
// CHECK-ERROR-NEXT:   In file included from {{.*}}/sycl-jit-toolchain//bin/../include/sycl/detail/core.hpp:21:
// CHECK-ERROR-NEXT:   In file included from {{.*}}/sycl-jit-toolchain//bin/../include/sycl/accessor.hpp:11:
// CHECK-ERROR-NEXT:   {{.*}}/sycl-jit-toolchain//bin/../include/sycl/access/access.hpp:14:10: fatal error: 'type_traits' file not found
// CHECK-ERROR-NEXT:      14 | #include <type_traits>
// CHECK-ERROR-NEXT:         |          ^~~~~~~~~~~~~

// Extra check that our in-memory libcxx/libc headers can really work on a
// system with no C/C++ toolchain:
// RUN: %{run} %t.out --sycl-rtc-in-memory-fs-only

// CUDA/HIP have SDK dependencies but exclude system includes so those aren't
// satisfied.
// REQUIRES: target-spir

#include <sycl/detail/core.hpp>
#include <sycl/kernel_bundle.hpp>
#include <sycl/usm.hpp>

namespace syclexp = sycl::ext::oneapi::experimental;

int main(int argc, char *argv[]) {
  sycl::queue q;
  std::string source = R"""(
    #include <sycl/sycl.hpp>
    namespace syclext = sycl::ext::oneapi;
    namespace syclexp = sycl::ext::oneapi::experimental;

    extern "C"
    SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::single_task_kernel))
    void foo(int *p) {
      *p = 42;
    }
)""";
  std::vector<std::string> opts;

  // Without this we see stack overflows on Win, but for some reason only in
  // `--sycl-rtc-in-memory-fs-only` mode when it should really be failing
  // earlier.
  opts.push_back("-fconstexpr-depth=128");

  for (int i = 1; i < argc; ++i)
    opts.emplace_back(argv[i]);
  try {

    auto kb_src = syclexp::create_kernel_bundle_from_source(
        q.get_context(), syclexp::source_language::sycl, source);
    auto kb_exe = syclexp::build(
        kb_src, syclexp::properties{syclexp::build_options{opts}});
    sycl::kernel krn = kb_exe.ext_oneapi_get_kernel("foo");
    auto *p = sycl::malloc_shared<int>(1, q);
    q.submit([&](sycl::handler &cgh) {
       cgh.set_args(p);
       cgh.single_task(krn);
     }).wait();
    std::cout << "Result: " << *p << std::endl;
    assert(*p == 42);
    sycl::free(p, q);
  } catch (const sycl::exception &e) {
    // Make `CHECK` lines more portable between Lin/Win:
    std::string s = e.what();
    std::replace(s.begin(), s.end(), '\\', '/');

    std::cout << s;
    return 1;
  }
}
