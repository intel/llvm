//==---- OsUtils.cpp --- os_utils unit test --------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <gtest/gtest.h>
#include <sycl/detail/os_util.hpp>

#ifdef _WIN32
/// Compare for string equality, but ignore difference between forward slash (/)
/// and backward slash (\).
///
/// This difference can be tricky to avoid, because CMake operates with forward
/// slashes even on Windows, and it can be problematic to convert them when
/// CMake generator expressions are involved. It's easier to handle slashes
/// here in the test itself.
bool isSameDir(const char* LHS, const char* RHS) {
  char L = 0, R = 0;
  do {
    L = *LHS++;
    R = *RHS++;
    if (L != R) {
      if (!((L == '\\' || L == '/') && (R == '\\' || R == '/'))) {
        return false;
      }
    }
  } while (L != '\0' && R != '\0');
  bool SameLen = (L == '\0' && R == '\0');
  return SameLen;
}
#else
#include <sys/stat.h>
#include <stdlib.h>
/// Check with respect to symbolic links
bool isSameDir(const char* LHS, const char* RHS) {
  struct stat StatBuf;
  if (stat(LHS, &StatBuf)) {
    perror("stat failed");
    return false;
  }
  ino_t InodeLHS = StatBuf.st_ino;
  if (stat(RHS, &StatBuf)) {
    perror("stat failed");
    return false;
  }
  ino_t InodeRHS = StatBuf.st_ino;
  return InodeRHS == InodeLHS;
}
#endif

class OsUtilsTest : public ::testing::Test {
};

TEST_F(OsUtilsTest, getCurrentDSODir) {
  std::string DSODir = sycl::detail::OSUtil::getCurrentDSODir();
  ASSERT_TRUE(isSameDir(DSODir.c_str(), SYCL_LIB_DIR)) <<
      "expected: " << SYCL_LIB_DIR << ", got: " << DSODir;
}
