// REQUIRES: (cpu || cuda ) && linux
// RUN: %clangxx -DSYCL_FALLBACK_ASSERT=1 -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// (see the other RUN lines below; it is a bit complicated)
//
// assert() call in device code guarantees nothing: on some devices it behaves
// in the usual way and terminates the program. On other devices it can print an
// error message and *continue* execution. Less capable devices can even ignore
// an assert!
//
// This makes testing an assert() a bit difficult task, and we have to rely on
// the implementation details to make sure that both "native" and "fallback"
// implementations work as expected.
//
// This test works only on Intel OpenCL CPU implementation, which is known to
// behave as follows:
//
//   Fallback mode (aka the best we can do by following the OpenCL spec):
//     1. Assertion condition is printed to *stdout* by the OpenCL printf().
//     2. Process (both host and device) is terminated by a SIGSEGV.
//
//   Native mode (same behavior as libc assert on CPU):
//     1. Assertion condition is printed to *stderr*.
//     2. Process (both host and device) is terminated by a SIGABRT.
//
// Other devices are "covered" by the assert-dummy.cpp test, which doesn't
// verify anything except a successful compilation for a device.
//
// FIXME: assert-dummy.cpp is not implemented yet, so other devices are not
// covered.
//
// How the test works:
// -------------------
//
//   1. First we verify that a call sequence in SYCL Runtime is correct:
//
//      - in the fallback mode we have to link an additional library that
//        provides a generic implementation of assert().
//
//      - in the native mode we don't link anything, and call clBuildProgram for
//        a user program alone.
//
//   2. Then we test that there is actually a difference between the two
//      modes. Since the CPU device is the only device that supports this
//      extension natively, we catch the difference between the fallback and the
//      native modes: SIGSEGV should occur in the fallback mode, SIGABRT in the
//      native mode.
//
//      In order to check the signal we fork() and let the child die. Then we
//      verify how it was terminated. EXPECTED_SIGNAL environment variable
//      controls the expected result.
//
//   3. We also test that a message is printed to the corresponding fd: stdout
//      for the fallback mode and stderr for the native mode. In the fallback
//      mode the test process dies right after a call to the OpenCL printf(), so
//      the message can still be buffered by stdio. We turn the bufferization
//      off explicitly.
//
//   4. We want to check both compilation flow in (1) and the message in (3),
//      but these messages can interleave and fail to match. To avoid this,
//      first run with SYCL_PI_TRACE and collect a trace, and then with
//      SHOULD_CRASH (without SYCL_PI_TRACE) to collect an error message.
//
// SYCL_DEVICELIB_INHIBIT_NATIVE=1 environment variable is used to force a mode
// in SYCL Runtime, so it doesn't look into a device extensions list and always
// link the fallback library.
//
//
// We also skip the native test entirely (see SKIP_IF_NO_EXT), since the assert
// extension is a new feature and may not be supported by the runtime used with
// SYCL.
//
// Overall this sounds stable enough. What could possibly go wrong?
//
// With either a CPU run or a GPU run we reset the output file and append the
// results of the runs. Otherwise a skipped GPU run may remove the output from
// a CPU run prior to running FileCheck.
// RUN: echo "" > %t.stderr.native
// RUN: %CPU_RUN_PLACEHOLDER SYCL_PI_TRACE=2 SHOULD_CRASH=1 EXPECTED_SIGNAL=SIGABRT %t.out 2>> %t.stderr.native
// RUN: %GPU_RUN_PLACEHOLDER SHOULD_CRASH=1 EXPECTED_SIGNAL=SIGIOT %t.out 2>> %t.stderr.native
// RUN: FileCheck %s --input-file %t.stderr.native --check-prefixes=CHECK-MESSAGE || FileCheck %s --input-file %t.stderr.native --check-prefix CHECK-NOTSUPPORTED
//
// Skip the test if the CPU RT doesn't support the extension yet:
// CHECK-NOTSUPPORTED: Device has no support for cl_intel_devicelib_assert
//
// Anyway, the same message has to be printed for both the fallback and the
// native modes (fallback prints to stdout, while native prints to stderr; we
// already handled this difference in the RUN lines):
//
// CHECK-MESSAGE: {{.*}}assert.cpp:{{[0-9]+}}: auto simple_vadd(const
// std::array<int, 3UL> &, const std::array<int, 3UL> &, std::array<int, 3UL>
// &)::(anonymous class)::operator()(sycl::handler &)::(anonymous
// class)::operator()(sycl::id<1>) const: global id: [{{[0-3]}},0,0], local
// id: [{{[0-3]}},0,0] Assertion `accessorC[wiID] == 0 && "Invalid value"`
// failed.
//
// Note that the work-item that hits the assert first may vary, since the order
// of execution is undefined. We catch only the first one (whatever id it is).

#include <array>
#include <assert.h>
#include <iostream>
#include <stdlib.h>
#include <sycl/sycl.hpp>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

using namespace sycl;

constexpr auto sycl_read = sycl::access::mode::read;
constexpr auto sycl_write = sycl::access::mode::write;

const int EXIT_SKIP_TEST = 42;

template <typename T, size_t N>
void simple_vadd(const std::array<T, N> &VA, const std::array<T, N> &VB,
                 std::array<T, N> &VC) {
  queue deviceQueue([](sycl::exception_list ExceptionList) {
    for (std::exception_ptr ExceptionPtr : ExceptionList) {
      try {
        std::rethrow_exception(ExceptionPtr);
      } catch (sycl::exception &E) {
        std::cerr << E.what() << std::endl;
      } catch (...) {
        std::cerr << "Unknown async exception was caught." << std::endl;
      }
    }
  });
  device dev = deviceQueue.get_device();
  bool unsupported = true;
  for (auto &ext : dev.get_info<info::device::extensions>()) {
    if (ext == "cl_intel_devicelib_assert") {
      unsupported = false;
    }
  }
  if (unsupported && getenv("SKIP_IF_NO_EXT")) {
    fprintf(stderr, "Device has no support for cl_intel_devicelib_assert, "
                    "skipping the test\n");
    exit(EXIT_SKIP_TEST);
  }

  sycl::range<1> numOfItems{N};
  sycl::buffer<T, 1> bufferA(VA.data(), numOfItems);
  sycl::buffer<T, 1> bufferB(VB.data(), numOfItems);
  sycl::buffer<T, 1> bufferC(VC.data(), numOfItems);

  deviceQueue.submit([&](sycl::handler &cgh) {
    auto accessorA = bufferA.template get_access<sycl_read>(cgh);
    auto accessorB = bufferB.template get_access<sycl_read>(cgh);
    auto accessorC = bufferC.template get_access<sycl_write>(cgh);

    cgh.parallel_for<class SimpleVaddT>(numOfItems, [=](sycl::id<1> wiID) {
      accessorC[wiID] = accessorA[wiID] + accessorB[wiID];
      assert(accessorC[wiID] == 0 && "Invalid value");
    });
  });
  deviceQueue.wait_and_throw();
}

int main() {
  int child = fork();
  if (child) {
    int status = 0;
    waitpid(child, &status, 0);
    if (WIFEXITED(status) && WEXITSTATUS(status) == EXIT_SKIP_TEST) {
      return 0;
    }
    if (getenv("SHOULD_CRASH")) {
      if (!WIFSIGNALED(status)) {
        fprintf(stderr, "error: process did not terminate by a signal\n");
        return 1;
      }
    } else {
      if (WIFSIGNALED(status)) {
        fprintf(stderr, "error: process should not terminate\n");
        return 1;
      }
      // We should not check anything if the child finished successful and this
      // was expected.
      return 0;
    }
    int sig = WTERMSIG(status);
    int expected = 0;
    if (const char *env = getenv("EXPECTED_SIGNAL")) {
      if (0 == strcmp(env, "SIGABRT")) {
        expected = SIGABRT;
      } else if (0 == strcmp(env, "SIGSEGV")) {
        expected = SIGSEGV;
      } else if (0 == strcmp(env, "SIGIOT")) {
        expected = SIGIOT;
      }
      if (!expected) {
        fprintf(stderr, "EXPECTED_SIGNAL should be set to either \"SIGABRT\", "
                        "or \"SIGSEGV\"!\n");
        return 1;
      }
    }
    if (sig != expected) {
      fprintf(stderr, "error: expected signal %d, got %d\n", expected, sig);
      return 1;
    }
    return 0;
  }

  // Turn the bufferization off to not loose the assert message if it is written
  // to stdout.
  if (setvbuf(stdout, NULL, _IONBF, 0)) {
    perror("failed to turn off bufferization on stdout");
    return 1;
  }

  std::array<int, 3> A = {1, 2, 3};
  std::array<int, 3> B = {1, 2, 3};
  std::array<int, 3> C = {0, 0, 0};

  simple_vadd(A, B, C);
}
