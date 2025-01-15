/*
 *
 * Copyright (C) 2022-2024 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
 * Exceptions. See LICENSE.TXT
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#include "ur_util.hpp"
#include "logger/ur_logger.hpp"

#ifdef _WIN32
#include <windows.h>
int ur_getpid(void) { return static_cast<int>(GetCurrentProcessId()); }

int ur_close_fd(int fd) { return -1; }

int ur_duplicate_fd(int pid, int fd_in) {
  // TODO: find another way to obtain a duplicate of another process's file
  // descriptor
  (void)pid;   // unused
  (void)fd_in; // unused
  return -1;
}

#else

#include <sys/syscall.h>
#include <unistd.h>
int ur_getpid(void) { return static_cast<int>(getpid()); }

int ur_close_fd(int fd) { return close(fd); }

int ur_duplicate_fd(int pid, int fd_in) {
// pidfd_getfd(2) is used to obtain a duplicate of another process's file
// descriptor. Permission to duplicate another process's file descriptor is
// governed by a ptrace access mode PTRACE_MODE_ATTACH_REALCREDS check (see
// ptrace(2)) that can be changed using the /proc/sys/kernel/yama/ptrace_scope
// interface. pidfd_getfd(2) is supported since Linux 5.6 pidfd_open(2) is
// supported since Linux 5.3
#if defined(__NR_pidfd_open) && defined(__NR_pidfd_getfd)
  errno = 0;
  int pid_fd = syscall(__NR_pidfd_open, pid, 0);
  if (pid_fd == -1) {
    logger::error("__NR_pidfd_open");
    return -1;
  }

  int fd_dup = syscall(__NR_pidfd_getfd, pid_fd, fd_in, 0);
  close(pid_fd);
  if (fd_dup == -1) {
    logger::error("__NR_pidfd_getfd");
    return -1;
  }

  return fd_dup;
#else
  // TODO: find another way to obtain a duplicate of another process's file
  // descriptor
  (void)pid;       // unused
  (void)fd_in;     // unused
  errno = ENOTSUP; // unsupported
  logger::error("__NR_pidfd_open or __NR_pidfd_getfd not available");
  return -1;
#endif /* defined(__NR_pidfd_open) && defined(__NR_pidfd_getfd) */
}

#endif /* _WIN32 */

std::optional<std::string> ur_getenv(const char *name) {
#if defined(_WIN32)
  constexpr int buffer_size = 1024;
  char buffer[buffer_size];
  auto rc = GetEnvironmentVariableA(name, buffer, buffer_size);
  if (0 != rc && rc < buffer_size) {
    return std::string(buffer);
  } else if (rc >= buffer_size) {
    std::stringstream ex_ss;
    ex_ss << "Environment variable " << name << " value too long!"
          << " Maximum length is " << buffer_size - 1 << " characters.";
    throw std::invalid_argument(ex_ss.str());
  }
  return std::nullopt;
#else
  const char *tmp_env = getenv(name);
  if (tmp_env != nullptr) {
    return std::string(tmp_env);
  } else {
    return std::nullopt;
  }
#endif
}
