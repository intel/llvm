//==---- OsUtils.cpp --- os_utils unit test --------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl/detail/os_util.hpp>
#include <gtest/gtest.h>

#ifdef __unix__
#include <sys/stat.h>
#include <stdlib.h>
/// Check with respect to symbolic links
bool isSameDir(const char* LHS, const char* RHS) {
  struct stat StatBuf;
  if (stat(LHS, &StatBuf)) {
    perror("stat failed");
    exit(EXIT_FAILURE);
  }
  ino_t InodeLHS = StatBuf.st_ino;
  if (stat(RHS, &StatBuf)) {
    perror("stat failed");
    exit(EXIT_FAILURE);
  }
  ino_t InodeRHS = StatBuf.st_ino;
  return InodeRHS == InodeLHS;
}
#else
bool isSameDir(const char* LHS, const char* RHS) {
  return 0 == strcmp(LHS, RHS);
}
#endif

class OsUtilsTest : public ::testing::Test {
};

TEST_F(OsUtilsTest, getCurrentDSODir) {
  std::string DSODir = cl::sycl::detail::OSUtil::getCurrentDSODir();
  ASSERT_TRUE(isSameDir(DSODir.c_str(), SYCL_LIB_DIR)) <<
      "expected: " << SYCL_LIB_DIR << ", got: " << DSODir;
}
