// A USM device allocation exported by a producer process is imported by a
// consumer process, and both processes access it from graphs. K is
// ConsumerReplays.
//
//   producer: graph {Data[i] = i}
//   consumer: graph {Data[i] += 1; copy Data} replayed K times, checks i + K
//   producer: eager copy Data, checks i + K
//
// With USE_ZE_IPC the IPC get/put/open/close calls are made directly through
// Level Zero instead of the SYCL IPC extension.

#include "../graph_common.hpp"

#include <sycl/ext/oneapi/experimental/ipc_memory.hpp>
#include <sycl/properties/queue_properties.hpp>

#if defined(__linux__)
#include <sys/prctl.h>
#include <unistd.h>
#endif // defined(__linux__)

namespace ipc = exp_ext::ipc;

#ifdef USE_ZE_IPC
#include "../ze_common.hpp"

// Level Zero equivalents of the ipc::memory functions.
namespace ze_ipc {
// The Level Zero IPC handle only describes the whole allocation, while a SYCL
// USM pointer may be a sub-allocation of a pooled allocation, so the offset
// into the allocation is sent alongside the handle.
struct handle {
  size_t Offset;
  ze_ipc_mem_handle_t ZeHandle;

  ipc::handle_data_t data() const {
    auto *Bytes = reinterpret_cast<const std::byte *>(this);
    return {Bytes, Bytes + sizeof(*this)};
  }
};

void *allocBase(void *Ptr, ze_context_handle_t ZeCtx) {
  void *Base = nullptr;
  ASSERT_ZE_RESULT_SUCCESS(zeMemGetAddressRange(ZeCtx, Ptr, &Base, nullptr));
  return Base;
}

handle get(void *Ptr, const context &Ctx) {
  auto ZeCtx = get_native<backend::ext_oneapi_level_zero>(Ctx);
  void *Base = allocBase(Ptr, ZeCtx);
  handle Handle{
      static_cast<size_t>(static_cast<char *>(Ptr) - static_cast<char *>(Base)),
      {}};
  ASSERT_ZE_RESULT_SUCCESS(zeMemGetIpcHandle(ZeCtx, Base, &Handle.ZeHandle));
  return Handle;
}

void put(handle &Handle, const context &Ctx) {
  ASSERT_ZE_RESULT_SUCCESS(zeMemPutIpcHandle(
      get_native<backend::ext_oneapi_level_zero>(Ctx), Handle.ZeHandle));
}

void *open(const ipc::handle_data_t &HandleData, const context &Ctx,
           const device &Dev) {
  assert(HandleData.size() == sizeof(handle));
  handle Handle;
  std::memcpy(&Handle, HandleData.data(), sizeof(Handle));

  // The driver duplicates the producer's file descriptor into this process
  // itself, so the handle is passed through unchanged.
  void *Base = nullptr;
  ASSERT_ZE_RESULT_SUCCESS(
      zeMemOpenIpcHandle(get_native<backend::ext_oneapi_level_zero>(Ctx),
                         get_native<backend::ext_oneapi_level_zero>(Dev),
                         Handle.ZeHandle, 0, &Base));
  return static_cast<char *>(Base) + Handle.Offset;
}

void close(void *Ptr, const context &Ctx) {
  auto ZeCtx = get_native<backend::ext_oneapi_level_zero>(Ctx);
  ASSERT_ZE_RESULT_SUCCESS(zeMemCloseIpcHandle(ZeCtx, allocBase(Ptr, ZeCtx)));
}
} // namespace ze_ipc

namespace ipc_memory = ze_ipc;
#else
namespace ipc_memory = ipc::memory;
#endif // USE_ZE_IPC

constexpr size_t N = 1024;
constexpr int ConsumerReplays = 3;

template <typename RecordFn>
void recordAndReplay(queue &Queue, int Replays, RecordFn Record) {
#ifdef GRAPH_E2E_NATIVE_RECORDING
  exp_ext::command_graph Graph{
      Queue.get_context(),
      Queue.get_device(),
      {exp_ext::property::graph::enable_native_recording{}}};
#else
  exp_ext::command_graph Graph{Queue.get_context(), Queue.get_device()};
#endif

  Graph.begin_recording(Queue);
  Record();
  Graph.end_recording(Queue);

  auto ExecGraph = Graph.finalize();
  for (int I = 0; I < Replays; ++I)
    Queue.ext_oneapi_graph(ExecGraph);
  Queue.wait_and_throw();
}

int countMismatches(const int *Got, const std::string &Name) {
  int Errors = 0;
  for (size_t I = 0; I < N; ++I)
    Errors +=
        !check_value(I, static_cast<int>(I) + ConsumerReplays, Got[I], Name);
  return Errors;
}

int producer(const std::string &Exe, const std::string &CommsFile) {
  queue Queue{property::queue::in_order{}};
  context Ctx = Queue.get_context();

  // UMF currently requires ptrace permissions to be set for the producer. As
  // such we need to set it until this limitation has been addressed.
  // https://github.com/oneapi-src/unified-memory-framework/tree/main?tab=readme-ov-file#level-zero-memory-provider
#if defined(__linux__)
  if (Queue.get_backend() == backend::ext_oneapi_level_zero &&
      prctl(PR_SET_PTRACER, getppid()) == -1) {
    std::cout << "Failed to set ptracer permissions!" << std::endl;
    return 1;
  }
#endif // defined(__linux__)

  int *Data = malloc_device<int>(N, Queue);
  int *Snapshot = malloc_host<int>(N, Queue);
  recordAndReplay(Queue, 1, [&] {
    Queue.parallel_for(range<1>{N},
                       [=](id<1> Idx) { Data[Idx] = static_cast<int>(Idx); });
  });

  auto Handle = ipc_memory::get(Data, Ctx);
  ipc::handle_data_t HandleData = Handle.data();
  std::ofstream{CommsFile, std::ios::binary}.write(
      reinterpret_cast<const char *>(HandleData.data()), HandleData.size());

  std::string Cmd = Exe + " consumer";
  std::cout << "Spawning: " << Cmd << "\n";
  int Rc = std::system(Cmd.c_str());
  int Errors = 0;
  if (Rc != 0) {
    std::cerr << "Consumer process failed, raw status " << Rc << "\n";
    Errors = 1;
  } else {
    // TODO: Read Data back in a graph. On BMG with the L0v2 adapter, a graph
    // in the exporting process that accesses Data after the consumer has
    // imported and closed it fails with UR_RESULT_ERROR_DEVICE_LOST or
    // UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY, while an eager copy works. This was
    // observed in CI on Linux kernel 6.11 but does not reproduce locally on
    // kernel 6.17 with the identical UMD.
    Queue.memcpy(Snapshot, Data, N * sizeof(int)).wait();
    Errors = countMismatches(Snapshot, "Producer");
  }

  ipc_memory::put(Handle, Ctx);
  std::remove(CommsFile.c_str());
  free(Snapshot, Queue);
  free(Data, Queue);
  return Errors != 0;
}

int consumer(const std::string &CommsFile) {
  queue Queue{property::queue::in_order{}};
  context Ctx = Queue.get_context();

  std::ifstream FS(CommsFile, std::ios::binary | std::ios::ate);
  if (!FS) {
    std::cerr << "Consumer failed to open " << CommsFile << "\n";
    return 1;
  }
  ipc::handle_data_t HandleData(FS.tellg());
  FS.seekg(0);
  FS.read(reinterpret_cast<char *>(HandleData.data()), HandleData.size());

  int *Data =
      static_cast<int *>(ipc_memory::open(HandleData, Ctx, Queue.get_device()));
  int *Snapshot = malloc_host<int>(N, Queue);

  recordAndReplay(Queue, ConsumerReplays, [&] {
    Queue.parallel_for(range<1>{N}, [=](id<1> Idx) { Data[Idx] += 1; });
    Queue.memcpy(Snapshot, Data, N * sizeof(int));
  });
  int Errors = countMismatches(Snapshot, "Consumer");

  free(Snapshot, Queue);
  ipc_memory::close(Data, Ctx);
  return Errors != 0;
}

int main(int argc, char *argv[]) {
  std::string Exe = argv[0];
  std::string CommsFile = Exe + ".ipc_handle";
  return argc == 1 ? producer(Exe, CommsFile) : consumer(CommsFile);
}
