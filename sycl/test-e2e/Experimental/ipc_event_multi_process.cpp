// REQUIRES: aspect-ext_oneapi_ipc_event
// REQUIRES: level_zero_v2_adapter
// REQUIRES: arch-intel_gpu_bmg_g21 || arch-intel_gpu_bmg_g31
// UNSUPPORTED: windows
// UNSUPPORTED-INTENDED: This test relies on POSIX
//   semantics (std::system + binary self-exec).

// RUN: %{build} -lze_loader %level_zero_options -o %t.out
// RUN: %{run} %t.out

// Cross-process IPC event round trip: producer spawns a consumer with
// std::system() and transports the IPC handle bytes through a file.

#include "Inputs/ipc_event_l0_signal.hpp"
#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/ipc_event.hpp>

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>

#if defined(__linux__)
#include <linux/prctl.h>
#include <sys/prctl.h>
#include <unistd.h>
#endif // defined(__linux__)

namespace exp = sycl::ext::oneapi::experimental;
namespace ipc = sycl::ext::oneapi::experimental::ipc;

constexpr const char *CommsFile = "ipc_event_comms.bin";

int spawner(int argc, char *argv[]) {
  assert(argc == 1);
  sycl::queue Q;
  sycl::device Dev = Q.get_device();
  sycl::context Ctx = Q.get_context();

#if defined(__linux__)
  // Allow the consumer to pidfd_getfd into this process.
  if (Q.get_backend() == sycl::backend::ext_oneapi_level_zero &&
      prctl(PR_SET_PTRACER, PR_SET_PTRACER_ANY) == -1) {
    std::cerr << "Failed to set ptracer permissions!\n";
    return 1;
  }
#endif

  sycl::event ProducerEvt =
      exp::make_event(Ctx, exp::properties{exp::enable_ipc});
  ipc_event_test::signalEventViaLevelZero(ProducerEvt, Ctx, Dev);

  ipc::handle Handle = ipc::event::get(ProducerEvt);
  ipc::handle_data_t Bytes = Handle.data();

  // Write the handle bytes; layout matches readHandleFile in
  // Inputs/ipc_event_sentinel.hpp: [size_t size][size bytes].
  {
    std::ofstream FS(CommsFile, std::ios::out | std::ios::binary);
    size_t HandleSize = Bytes.size();
    FS.write(reinterpret_cast<const char *>(&HandleSize), sizeof(size_t));
    FS.write(reinterpret_cast<const char *>(Bytes.data()), HandleSize);
  }

  std::string Cmd = std::string{argv[0]} + " consumer";
  std::cout << "Spawning: " << Cmd << "\n";
  int Rc = std::system(Cmd.c_str());

  ipc::event::put(Handle, Ctx);
  std::remove(CommsFile);

  if (!WIFEXITED(Rc)) {
    std::cerr << "Consumer process terminated abnormally, raw status " << Rc
              << "\n";
    return 2;
  }
  if (WEXITSTATUS(Rc) != 0) {
    std::cerr << "Consumer process failed with exit code " << WEXITSTATUS(Rc)
              << "\n";
    return 2;
  }
  return 0;
}

int consumer() {
  sycl::queue Q;
  sycl::device Dev = Q.get_device();
  sycl::context Ctx = Q.get_context();

  std::ifstream FS(CommsFile, std::ios::in | std::ios::binary);
  if (!FS) {
    std::cerr << "consumer: failed to open " << CommsFile << "\n";
    return 1;
  }
  size_t HandleSize = 0;
  FS.read(reinterpret_cast<char *>(&HandleSize), sizeof(size_t));
  if (HandleSize == 0) {
    std::cerr << "consumer: zero-sized handle\n";
    return 2;
  }
  std::unique_ptr<std::byte[]> HandleBuffer{new std::byte[HandleSize]};
  FS.read(reinterpret_cast<char *>(HandleBuffer.get()), HandleSize);

  ipc::handle_data_t HandleVec(HandleBuffer.get(),
                               HandleBuffer.get() + HandleSize);

  sycl::event Imported = ipc::event::open(HandleVec, Ctx);
  Imported.wait();
  return 0;
}

int main(int argc, char *argv[]) {
  return argc == 1 ? spawner(argc, argv) : consumer();
}
