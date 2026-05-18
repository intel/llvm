// REQUIRES: aspect-usm_device_allocations && aspect-ext_oneapi_ipc_memory

// DEFINE: %{cpp20} = %if cl_options %{/clang:-std=c++20%} %else %{-std=c++20%}

// RUN: %{build} -o %t.out
// RUN: %{run-unfiltered-devices} %t.out
// RUN: %{build} -DUSE_VIEW %{cpp20} -o %t.view.out
// RUN: %{run-unfiltered-devices} %t.view.out

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/ipc_memory.hpp>
#include <sycl/usm.hpp>

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

#if defined(__linux__)
#include <linux/prctl.h>
#include <sys/prctl.h>
#include <unistd.h>
#elif defined(__WIN32__) || defined(_WIN32)
#include <windows.h>
#endif // defined(__linux__)

namespace syclexp = sycl::ext::oneapi::experimental;

constexpr size_t N = 32;
constexpr const char *CommsFile = "ipc_comms.txt";

static void print_env(const char *Name) {
  const char *Value = std::getenv(Name);
  std::cout << Name << '=' << (Value ? Value : "<unset>") << std::endl;
}

static void set_env(const char *Name, const char *Value) {
#if defined(__WIN32__) || defined(_WIN32)
  if (!SetEnvironmentVariableA(Name, Value)) {
    std::cout << "SetEnvironmentVariableA failed for " << Name << ": "
              << GetLastError() << std::endl;
    throw std::runtime_error("SetEnvironmentVariableA failed");
  }
#else
  if (setenv(Name, Value, 1) != 0) {
    throw std::runtime_error("setenv failed");
  }
#endif
}

static void configure_runtime_diagnostics_env() {
  set_env("ONEAPI_DEVICE_SELECTOR", "level_zero:gpu");
  set_env("UR_L0_V2_ENABLE_WINDOWS_IPC_WA", "1");
  set_env("SYCL_UR_TRACE", "-1");
  set_env("UR_LOG_LOADER", "level:debug;output:stdout;flush:debug");
  set_env("UR_LOG_LEVEL_ZERO", "level:debug;output:stdout;flush:debug");
  set_env("UMF_LOG", "level:debug;flush:debug;output:stdout;pid:yes");
}

static void print_runtime_diagnostics(const char *Role, sycl::queue &Q) {
  std::cout << '[' << Role << "] backend=" << static_cast<int>(Q.get_backend())
            << " device=" << Q.get_device().get_info<sycl::info::device::name>()
            << std::endl;
  print_env("ONEAPI_DEVICE_SELECTOR");
  print_env("UR_L0_V2_ENABLE_WINDOWS_IPC_WA");
  print_env("SYCL_UR_TRACE");
  print_env("UR_LOG_LOADER");
  print_env("UR_LOG_LEVEL_ZERO");
  print_env("UMF_LOG");
}

void spawn_and_sync(std::string Exe) {
  std::string Cmd = '"' + Exe + '"' + " 1";
  std::cout << "Spawning: " << Cmd << std::endl;
#if defined(__WIN32__) || defined(_WIN32)
  STARTUPINFO StartupInfo;
  PROCESS_INFORMATION ProcInfo;

  std::memset(&ProcInfo, 0, sizeof(ProcInfo));
  std::memset(&StartupInfo, 0, sizeof(StartupInfo));
  StartupInfo.cb = sizeof(StartupInfo);
  BOOL Created =
      CreateProcessA(NULL, const_cast<char *>(Cmd.c_str()), NULL, NULL, TRUE, 0,
                     NULL, NULL, &StartupInfo, &ProcInfo);
  std::cout << "CreateProcessA result: " << Created << std::endl;
  if (!Created) {
    std::cout << "CreateProcessA GetLastError: " << GetLastError() << std::endl;
    throw std::runtime_error("CreateProcessA failed");
  }

  DWORD WaitStatus = WaitForSingleObject(ProcInfo.hProcess, 30000);
  std::cout << "WaitForSingleObject result: " << WaitStatus << std::endl;
  if (WaitStatus == WAIT_FAILED) {
    std::cout << "WaitForSingleObject GetLastError: " << GetLastError()
              << std::endl;
    CloseHandle(ProcInfo.hProcess);
    CloseHandle(ProcInfo.hThread);
    throw std::runtime_error("WaitForSingleObject failed");
  }

  if (WaitStatus == WAIT_TIMEOUT) {
    CloseHandle(ProcInfo.hProcess);
    CloseHandle(ProcInfo.hThread);
    throw std::runtime_error("Child process timed out");
  }

  DWORD ExitCode = 0;
  if (!GetExitCodeProcess(ProcInfo.hProcess, &ExitCode)) {
    std::cout << "GetExitCodeProcess GetLastError: " << GetLastError()
              << std::endl;
    CloseHandle(ProcInfo.hProcess);
    CloseHandle(ProcInfo.hThread);
    throw std::runtime_error("GetExitCodeProcess failed");
  }
  std::cout << "Child exit code: " << ExitCode << std::endl;
  CloseHandle(ProcInfo.hProcess);
  CloseHandle(ProcInfo.hThread);
  if (ExitCode != 0)
    throw std::runtime_error("Child process returned non-zero exit code");
#else
  std::system(Cmd.c_str());
#endif
}

int spawner(int argc, char *argv[]) try {
  std::cout << "Running spanwer..." << std::endl;
  assert(argc == 1);
  configure_runtime_diagnostics_env();
  sycl::queue Q;
  print_runtime_diagnostics("spawner", Q);

#if defined(__linux__)
  // UMF currently requires ptrace permissions to be set for the spawner. As
  // such we need to set it until this limitation has been addressed.
  // https://github.com/oneapi-src/unified-memory-framework/tree/main?tab=readme-ov-file#level-zero-memory-provider
  if (Q.get_backend() == sycl::backend::ext_oneapi_level_zero &&
      prctl(PR_SET_PTRACER, getppid()) == -1) {
    std::cout << "Failed to set ptracer permissions!" << std::endl;
    return 1;
  }
#endif // defined(__linux__)

  int *DataPtr = sycl::malloc_device<int>(N, Q);
  Q.parallel_for(N, [=](sycl::item<1> I) {
     DataPtr[I] = static_cast<int>(I.get_linear_id());
   }).wait();

  {
    // Write handle data to file.
    {
      syclexp::ipc_memory::handle Handle =
          syclexp::ipc_memory::get(DataPtr, Q.get_context());
#ifdef USE_VIEW
      syclexp::ipc_memory::handle_data_view_t HandleData = Handle.data_view();
#else
      syclexp::ipc_memory::handle_data_t HandleData = Handle.data();
#endif
      size_t HandleDataSize = HandleData.size();
      std::cout << "Spawner handle size: " << HandleDataSize << std::endl;
      std::fstream FS(CommsFile, std::ios_base::out | std::ios_base::binary);
      FS.write(reinterpret_cast<const char *>(&HandleDataSize), sizeof(size_t));
      FS.write(reinterpret_cast<const char *>(HandleData.data()),
               HandleDataSize);
    }

    // Spawn other process with an argument.
    spawn_and_sync(std::string{argv[0]});
  }

  int Failures = 0;
  int Read[N] = {0};
  Q.copy(DataPtr, Read, N).wait();
  for (size_t I = 0; I < N; ++I) {
    if (Read[I] != (N - I)) {
      ++Failures;
      std::cout << "Failed from spawner: Result at " << I
                << " unexpected: " << Read[I] << " != " << (N - I) << std::endl;
    }
  }
  sycl::free(DataPtr, Q);
  return Failures;
} catch (sycl::exception &e) {
  std::cout << "Spawner failed: " << e.what() << std::endl;
  throw;
}

int consumer() try {
  std::cout << "Running consumer..." << std::endl;
  configure_runtime_diagnostics_env();
  sycl::queue Q;
  print_runtime_diagnostics("consumer", Q);

  // Read the handle data.
  std::fstream FS(CommsFile, std::ios_base::in | std::ios_base::binary);
  size_t HandleSize = 0;
  FS.read(reinterpret_cast<char *>(&HandleSize), sizeof(size_t));
  std::cout << "Consumer handle size: " << HandleSize << std::endl;
  std::unique_ptr<std::byte[]> HandleData{new std::byte[HandleSize]};
  FS.read(reinterpret_cast<char *>(HandleData.get()), HandleSize);

  // Open IPC handle.
#ifdef USE_VIEW
  syclexp::ipc_memory::handle_data_view_t Handle{HandleData.get(), HandleSize};
#else
  syclexp::ipc_memory::handle_data_t Handle{HandleData.get(),
                                            HandleData.get() + HandleSize};
#endif
  int *DataPtr = reinterpret_cast<int *>(
      syclexp::ipc_memory::open(Handle, Q.get_context(), Q.get_device()));
  std::cout << "Consumer open succeeded: " << static_cast<void *>(DataPtr)
            << std::endl;

  // Test the data already in the USM pointer.
  int Failures = 0;
  int Read[N] = {0};
  Q.copy(DataPtr, Read, N).wait();
  for (size_t I = 0; I < N; ++I) {
    if (Read[I] != I) {
      ++Failures;
      std::cout << "Failed from consumer: Result at " << I
                << " unexpected: " << Read[I] << " != " << I << std::endl;
    }
  }

  Q.parallel_for(N, [=](sycl::item<1> I) {
     DataPtr[I] = static_cast<int>(N - I.get_linear_id());
   }).wait();

  // Close the IPC pointer.
  syclexp::ipc_memory::close(DataPtr, Q.get_context());
  std::cout << "Consumer close succeeded" << std::endl;

  return Failures;
} catch (sycl::exception &e) {
  std::cout << "Consumer failed: " << e.what() << std::endl;
  throw;
}

int main(int argc, char *argv[]) {
  return argc == 1 ? spawner(argc, argv) : consumer();
}
