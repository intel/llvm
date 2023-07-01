//==- DeviceConfigFileAspects.cpp --- Device config file aspects unit test -==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
#include <map>

#include <llvm/ADT/StringRef.h>
#include <llvm/SYCLLowerIR/DeviceConfigFile.hpp>
#include <sycl/sycl.hpp>

#include <helpers/PiMock.hpp>

#include <gtest/gtest.h>

#define __SYCL_ASPECT_DEPRECATED_ALIAS(ASPECT, ID, MESSAGE)                    \
  __SYCL_ASPECT_DEPRECATED(ASPECT, ID, MESSAGE)

TEST(DeviceConfigFile, DeviceConfigFileAspects) {
  auto testAspects = DeviceConfigFile::TargetTable.find("__TestAspectList");
  assert(testAspects != DeviceConfigFile::TargetTable.end());
  auto aspectsList = testAspects->second.aspects;

#define __SYCL_ASPECT(ASPECT, ASPECT_VAL)                                      \
  llvm::StringRef s##ASPECT(#ASPECT);                                          \
  EXPECT_TRUE(std::find(aspectsList.begin(), aspectsList.end(), s##ASPECT) !=  \
              aspectsList.end());

#include <sycl/info/aspects.def>

#undef __SYCL_ASPECT

  auto testDeprecatedAspects =
      DeviceConfigFile::TargetTable.find("__TestDeprecatedAspectList");
  assert(testDeprecatedAspects != DeviceConfigFile::TargetTable.end());
  auto deprecatedAspectsList = testDeprecatedAspects->second.aspects;
#define __SYCL_ASPECT_DEPRECATED(ASPECT, ASPECT_VAL, MSG)                      \
  llvm::StringRef s##ASPECT(#ASPECT);                                          \
  EXPECT_TRUE(std::find(deprecatedAspectsList.begin(),                         \
                        deprecatedAspectsList.end(),                           \
                        s##ASPECT) != deprecatedAspectsList.end());

#include <sycl/info/aspects_deprecated.def>

#undef __SYCL_ASPECT_DEPRECATED
}

#undef __SYCL_ASPECT_DEPRECATED_ALIAS
