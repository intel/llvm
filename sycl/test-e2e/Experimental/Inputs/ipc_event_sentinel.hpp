// File-based rendezvous helpers shared by the cross-process IPC event tests.
// Paths are relative to the working directory; each test should prefix its
// file names to avoid collisions with other tests.

#pragma once

#include <sycl/ext/oneapi/experimental/ipc_event.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <initializer_list>
#include <memory>
#include <string>
#include <thread>

namespace ipc_event_test {

namespace ipc = sycl::ext::oneapi::experimental::ipc;

inline void waitForFile(const std::string &Path, int TimeoutSecs = 30) {
  for (int i = 0; i < TimeoutSecs * 100; ++i) {
    if (std::ifstream{Path}.good())
      return;
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  std::fprintf(stderr, "Timeout waiting for %s\n", Path.c_str());
  std::exit(1);
}

inline void touchFile(const std::string &Path) {
  std::ofstream F(Path);
  if (!F)
    std::fprintf(stderr, "Failed to create %s\n", Path.c_str());
}

// Remove any leftover sentinel files from a previous run so that waitForFile
// does not match a stale file. Call this in the producer before it spawns the
// consumer; missing files are ignored. Paths are relative to the working
// directory, which lit does not clean between reruns.
inline void removeStaleFiles(std::initializer_list<const char *> Paths) {
  for (const char *P : Paths)
    std::remove(P);
}

// Layout: [size_t size][size bytes].
inline void writeHandleFile(const std::string &Path, const ipc::handle &H) {
  ipc::handle_data_t Bytes = H.data();
  size_t Sz = Bytes.size();
  std::ofstream FS(Path, std::ios::out | std::ios::binary);
  FS.write(reinterpret_cast<const char *>(&Sz), sizeof(Sz));
  FS.write(reinterpret_cast<const char *>(Bytes.data()), Sz);
  if (!FS) {
    std::fprintf(stderr, "Failed to write handle to %s\n", Path.c_str());
    std::exit(1);
  }
}

inline ipc::handle_data_t readHandleFile(const std::string &Path) {
  std::ifstream FS(Path, std::ios::in | std::ios::binary);
  if (!FS) {
    std::fprintf(stderr, "Failed to open handle file %s\n", Path.c_str());
    std::exit(1);
  }
  size_t Sz = 0;
  FS.read(reinterpret_cast<char *>(&Sz), sizeof(Sz));
  std::unique_ptr<std::byte[]> Buf{new std::byte[Sz]};
  FS.read(reinterpret_cast<char *>(Buf.get()), Sz);
  return ipc::handle_data_t(Buf.get(), Buf.get() + Sz);
}

} // namespace ipc_event_test
