//==------- ConfigTests.cpp --- SYCL config processing unit test -----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/config.hpp>
#include <gtest/gtest.h>
#include <regex>

TEST(ConfigTests, CheckSpaceAtFirstPosition) {
  std::ofstream file("conf.txt");
  if (file.is_open()) {
    file << " a=b" << std::endl;
    file.close();
  }
  setenv("SYCL_CONFIG_FILE_NAME", "conf.txt", 1);
  try {
    sycl::detail::SYCLConfig<sycl::detail::SYCL_DEVICE_ALLOWLIST>::get();
    throw std::logic_error("sycl::exception didn't throw");
  } catch (sycl::exception &e) {
    EXPECT_EQ(
        std::string(
            "SPACE found at the beginning/end of the line or before/after '='"),
        e.what());
  } catch (...) {
    FAIL() << "Expected sycl::exception";
  }
}

TEST(ConfigTests, CheckSpaceAtLastPosition) {
  std::ofstream file("conf.txt");
  if (file.is_open()) {
    file << "a=b " << std::endl;
    file.close();
  }
  setenv("SYCL_CONFIG_FILE_NAME", "conf.txt", 1);
  try {
    sycl::detail::SYCLConfig<sycl::detail::SYCL_DEVICE_ALLOWLIST>::get();
    throw std::logic_error("sycl::exception didn't throw");
  } catch (sycl::exception &e) {
    EXPECT_EQ(
        std::string(
            "SPACE found at the beginning/end of the line or before/after '='"),
        e.what());
  } catch (...) {
    FAIL() << "Expected sycl::exception";
  }
}

TEST(ConfigTests, CheckSpaceBeforeAssignment) {
  std::ofstream file("conf.txt");
  if (file.is_open()) {
    file << "a =b" << std::endl;
    file.close();
  }
  setenv("SYCL_CONFIG_FILE_NAME", "conf.txt", 1);
  try {
    sycl::detail::SYCLConfig<sycl::detail::SYCL_DEVICE_ALLOWLIST>::get();
    throw std::logic_error("sycl::exception didn't throw");
  } catch (sycl::exception &e) {
    EXPECT_EQ(
        std::string(
            "SPACE found at the beginning/end of the line or before/after '='"),
        e.what());
  } catch (...) {
    FAIL() << "Expected sycl::exception";
  }
}

TEST(ConfigTests, CheckSpaceAfterAssignment) {
  std::ofstream file("conf.txt");
  if (file.is_open()) {
    file << "a= b" << std::endl;
    file.close();
  }
  setenv("SYCL_CONFIG_FILE_NAME", "conf.txt", 1);
  try {
    sycl::detail::SYCLConfig<sycl::detail::SYCL_DEVICE_ALLOWLIST>::get();
    throw std::logic_error("sycl::exception didn't throw");
  } catch (sycl::exception &e) {
    EXPECT_EQ(
        std::string(
            "SPACE found at the beginning/end of the line or before/after '='"),
        e.what());
  } catch (...) {
    FAIL() << "Expected sycl::exception";
  }
}

TEST(ConfigTests, CheckVariableNameBiggerThanMaxConfigName) {
  std::ofstream file("conf.txt");
  if (file.is_open()) {
    for (int i = 0; i <= 256; i++) {
      file << "a";
    }
    file << "=b" << std::endl;
    file.close();
  }
  setenv("SYCL_CONFIG_FILE_NAME", "conf.txt", 1);
  try {
    sycl::detail::SYCLConfig<sycl::detail::SYCL_DEVICE_ALLOWLIST>::get();
    throw std::logic_error("sycl::exception didn't throw");
  } catch (sycl::exception &e) {
    EXPECT_TRUE(std::regex_match(
        e.what(),
        std::regex(
            "Variable name is more than ([\\d]+) or less than one character")));
  } catch (...) {
    FAIL() << "Expected sycl::exception";
  }
}

TEST(ConfigTests, CheckVariableWithoutName) {
  std::ofstream file("conf.txt");
  if (file.is_open()) {
    file << "=b" << std::endl;
    file.close();
  }
  setenv("SYCL_CONFIG_FILE_NAME", "conf.txt", 1);
  try {
    sycl::detail::SYCLConfig<sycl::detail::SYCL_DEVICE_ALLOWLIST>::get();
    throw std::logic_error("sycl::exception didn't throw");
  } catch (sycl::exception &e) {
    EXPECT_TRUE(std::regex_match(
        e.what(),
        std::regex(
            "Variable name is more than ([\\d]+) or less than one character")));
  } catch (...) {
    FAIL() << "Expected sycl::exception";
  }
}

TEST(ConfigTests, CheckVariableValueBiggerThanMaxConfigValue) {
  std::ofstream file("conf.txt");
  if (file.is_open()) {
    file << "a=";
    for (int i = 0; i <= 1024; i++) {
      file << "b";
    }
    file << std::endl;
    file.close();
  }
  setenv("SYCL_CONFIG_FILE_NAME", "conf.txt", 1);
  try {
    sycl::detail::SYCLConfig<sycl::detail::SYCL_DEVICE_ALLOWLIST>::get();
    throw std::logic_error("sycl::exception didn't throw");
  } catch (sycl::exception &e) {
    EXPECT_TRUE(std::regex_match(
        e.what(), std::regex("The value contains more than ([\\d]+) characters "
                             "or does not contain them at all")));
  } catch (...) {
    FAIL() << "Expected sycl::exception";
  }
}

TEST(ConfigTests, CheckVariableWithoutValue) {
  std::ofstream file("conf.txt");
  if (file.is_open()) {
    file << "a=" << std::endl;
    file.close();
  }
  setenv("SYCL_CONFIG_FILE_NAME", "conf.txt", 1);
  try {
    sycl::detail::SYCLConfig<sycl::detail::SYCL_DEVICE_ALLOWLIST>::get();
    throw std::logic_error("sycl::exception didn't throw");
  } catch (sycl::exception &e) {
    EXPECT_TRUE(std::regex_match(
        e.what(), std::regex("The value contains more than ([\\d]+) characters "
                             "or does not contain them at all")));
  } catch (...) {
    FAIL() << "Expected sycl::exception";
  }
}

TEST(ConfigTests, CheckIncorrectComplexConfigProcessing) {
  std::ofstream file("conf.txt");
  if (file.is_open()) {
    file << "#\n   #\r\n#a=b\n\n\na\r\n aaa \r\na=b\r\na=\r" << std::endl;
    file.close();
  }
  setenv("SYCL_CONFIG_FILE_NAME", "conf.txt", 1);
  try {
    sycl::detail::SYCLConfig<sycl::detail::SYCL_DEVICE_ALLOWLIST>::get();
    throw std::logic_error("sycl::exception didn't throw");
  } catch (sycl::exception &e) {
    EXPECT_TRUE(std::regex_match(
        e.what(), std::regex("The value contains more than ([\\d]+) characters "
                             "or does not contain them at all")));
  } catch (...) {
    FAIL() << "Expected sycl::exception";
  }
}

TEST(ConfigTests, CheckSimpleCorrectConfigProcessing) {
  std::ofstream file("conf.txt");
  if (file.is_open()) {
    file << "SYCL_DEVICE_ALLOWLIST=1" << std::endl;
    file.close();
  }
  setenv("SYCL_CONFIG_FILE_NAME", "conf.txt", 1);
  sycl::detail::readConfig(true);
  EXPECT_EQ(
      std::string("1"),
      sycl::detail::SYCLConfig<sycl::detail::SYCL_DEVICE_ALLOWLIST>::get());
}

TEST(ConfigTests, CheckCorrectConfigProcessing) {
  std::ofstream file("conf.txt");
  if (file.is_open()) {
    file
        << "#a\r\n # b \r\n #a=b\r\n # a = b\r\n\r\nSYCL_DEVICE_ALLOWLIST=2\r\n"
        << std::endl;
    file.close();
  }
  setenv("SYCL_CONFIG_FILE_NAME", "conf.txt", 1);
  sycl::detail::readConfig(true);
  EXPECT_EQ(
      std::string("2"),
      sycl::detail::SYCLConfig<sycl::detail::SYCL_DEVICE_ALLOWLIST>::get());
}

TEST(ConfigTests, CheckMultiAssignment) {
  std::ofstream file("conf.txt");
  if (file.is_open()) {
    file << "a=b\r\nb=c\nSYCL_DEVICE_ALLOWLIST=3\r\nc=d";
    file.close();
  }
  setenv("SYCL_CONFIG_FILE_NAME", "conf.txt", 1);
  sycl::detail::readConfig(true);
  EXPECT_EQ(
      std::string("3"),
      sycl::detail::SYCLConfig<sycl::detail::SYCL_DEVICE_ALLOWLIST>::get());
}

TEST(ConfigTests, CheckAssignmentWithComment) {
  std::ofstream file("conf.txt");
  if (file.is_open()) {
    file << "SYCL_DEVICE_ALLOWLIST=4 #comment\r\n";
    file.close();
  }
  setenv("SYCL_CONFIG_FILE_NAME", "conf.txt", 1);
  sycl::detail::readConfig(true);
  EXPECT_EQ(
      std::string("4"),
      sycl::detail::SYCLConfig<sycl::detail::SYCL_DEVICE_ALLOWLIST>::get());
}
