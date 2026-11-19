// A kernel launched from a TU compiled with -fsycl-host-only has no device
// image in the program. The runtime must report that with a catchable
// sycl::exception rather than an assert or a crash.
//
// RUN: %{build} -fsycl-host-only -c -o %t.hostonly.o -DHOST_ONLY_TU
// RUN: %{build} -c -o %t.main.o -DMAIN_TU
// RUN: %clangxx -fsycl %{sycl_target_opts} %t.hostonly.o %t.main.o -Wno-unused-command-line-argument -o %t.out
// RUN: %{run} %t.out

#include <sycl/detail/core.hpp>

#include <cstdio>
#include <string>

#ifdef HOST_ONLY_TU
void launchFromHostOnlyTU(sycl::queue &Q) {
  Q.single_task([] {}).wait();
}
#endif

#ifdef MAIN_TU
void launchFromHostOnlyTU(sycl::queue &Q);

int main() {
  sycl::queue Q;
  Q.single_task([] {}).wait(); // this TU's own kernel is fine

  try {
    launchFromHostOnlyTU(Q);
  } catch (const sycl::exception &E) {
    bool Ok =
        E.code() == sycl::make_error_code(sycl::errc::runtime) &&
        std::string(E.what()).find("No kernel named") != std::string::npos;
    if (!Ok)
      std::printf("unexpected sycl::exception: %s\n", E.what());
    return Ok ? 0 : 1;
  }
  std::printf("expected a sycl::exception, none thrown\n");
  return 1;
}
#endif
