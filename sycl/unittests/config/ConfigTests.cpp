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

TEST(ConfigTests, DISABLED_CheckConfigProcessing) {
#ifdef _WIN32
  _putenv_s("SYCL_CONFIG_FILE_NAME", "conf.txt");
#else
  setenv("SYCL_CONFIG_FILE_NAME", "conf.txt", 1);
#endif

  // Check SPACE at first position
  std::ofstream File;
  File.open("conf.txt");
  if (File.is_open()) {
    File << " a=b" << std::endl;
    File.close();
  }
  try {
    sycl::detail::SYCLConfig<sycl::detail::SYCL_DEVICE_ALLOWLIST>::get();
    throw std::logic_error("sycl::exception didn't throw");
  } catch (sycl::exception &e) {
    EXPECT_EQ(
        std::string(
            "SPACE found at the beginning/end of the line or before/after '='"),
        e.what());
  } catch (...) {
    FAIL() << "Check SPACE at first position failed";
  }

  // Check SPACE at last position
  File.open("conf.txt");
  if (File.is_open()) {
    File << "a=b " << std::endl;
    File.close();
  }
  try {
    sycl::detail::SYCLConfig<sycl::detail::SYCL_DEVICE_ALLOWLIST>::get();
    throw std::logic_error("sycl::exception didn't throw");
  } catch (sycl::exception &e) {
    EXPECT_EQ(
        std::string(
            "SPACE found at the beginning/end of the line or before/after '='"),
        e.what());
  } catch (...) {
    FAIL() << "Check SPACE at last position failed";
  }

  // Check SPACE before assignment
  File.open("conf.txt");
  if (File.is_open()) {
    File << "a =b" << std::endl;
    File.close();
  }
  try {
    sycl::detail::SYCLConfig<sycl::detail::SYCL_DEVICE_ALLOWLIST>::get();
    throw std::logic_error("sycl::exception didn't throw");
  } catch (sycl::exception &e) {
    EXPECT_EQ(
        std::string(
            "SPACE found at the beginning/end of the line or before/after '='"),
        e.what());
  } catch (...) {
    FAIL() << "Check SPACE before assignment failed";
  }

  // Check SPACE after assignment
  File.open("conf.txt");
  if (File.is_open()) {
    File << "a= b" << std::endl;
    File.close();
  }
  try {
    sycl::detail::SYCLConfig<sycl::detail::SYCL_DEVICE_ALLOWLIST>::get();
    throw std::logic_error("sycl::exception didn't throw");
  } catch (sycl::exception &e) {
    EXPECT_EQ(
        std::string(
            "SPACE found at the beginning/end of the line or before/after '='"),
        e.what());
  } catch (...) {
    FAIL() << "Check SPACE after assignment failed";
  }

  // Check variable name bigger than MAX_CONFIG_NAME
  File.open("conf.txt");
  if (File.is_open()) {
    for (int i = 0; i <= sycl::detail::MAX_CONFIG_NAME; i++) {
      File << "a";
    }
    File << "=b" << std::endl;
    File.close();
  }
  try {
    sycl::detail::SYCLConfig<sycl::detail::SYCL_DEVICE_ALLOWLIST>::get();
    throw std::logic_error("sycl::exception didn't throw");
  } catch (sycl::exception &e) {
    EXPECT_TRUE(std::regex_match(
        e.what(),
        std::regex(
            "Variable name is more than ([\\d]+) or less than one character")));
  } catch (...) {
    FAIL() << "Check variable name bigger than MAX_CONFIG_NAME failed";
  }

  // Check variable without name
  File.open("conf.txt");
  if (File.is_open()) {
    File << "=b" << std::endl;
    File.close();
  }
  try {
    sycl::detail::SYCLConfig<sycl::detail::SYCL_DEVICE_ALLOWLIST>::get();
    throw std::logic_error("sycl::exception didn't throw");
  } catch (sycl::exception &e) {
    EXPECT_TRUE(std::regex_match(
        e.what(), std::regex("Variable name is more than ([\\d]+) or less "
                             "than one character")));
  } catch (...) {
    FAIL() << "Check variable without name failed";
  }

  // Check variable value bigger than MAX_CONFIG_VALUE
  File.open("conf.txt");
  if (File.is_open()) {
    File << "a=";
    for (int i = 0; i <= sycl::detail::MAX_CONFIG_VALUE; i++) {
      File << "b";
    }
    File << std::endl;
    File.close();
  }
  try {
    sycl::detail::SYCLConfig<sycl::detail::SYCL_DEVICE_ALLOWLIST>::get();
    throw std::logic_error("sycl::exception didn't throw");
  } catch (sycl::exception &e) {
    EXPECT_TRUE(std::regex_match(
        e.what(), std::regex("The value contains more than ([\\d]+) characters "
                             "or does not contain them at all")));
  } catch (...) {
    FAIL() << "Check variable value bigger than MAX_CONFIG_VALUE failed";
  }

  // Check variable without value
  File.open("conf.txt");
  if (File.is_open()) {
    File << "a=" << std::endl;
    File.close();
  }
  try {
    sycl::detail::SYCLConfig<sycl::detail::SYCL_DEVICE_ALLOWLIST>::get();
    throw std::logic_error("sycl::exception didn't throw");
  } catch (sycl::exception &e) {
    EXPECT_TRUE(std::regex_match(
        e.what(), std::regex("The value contains more than ([\\d]+) characters "
                             "or does not contain them at all")));
  } catch (...) {
    FAIL() << "Check variable without value failed";
  }

  // Check incorrect complex config processing
  File.open("conf.txt");
  if (File.is_open()) {
    File << "#\n   #\r\n#a=b\n\n\na\r\n aaa \r\na=b\r\na=\r" << std::endl;
    File.close();
  }
  try {
    sycl::detail::SYCLConfig<sycl::detail::SYCL_DEVICE_ALLOWLIST>::get();
    throw std::logic_error("sycl::exception didn't throw");
  } catch (sycl::exception &e) {
    EXPECT_TRUE(std::regex_match(
        e.what(), std::regex("The value contains more than ([\\d]+) characters "
                             "or does not contain them at all")));
  } catch (...) {
    FAIL() << "Check incorrect complex config processing failed";
  }

  // Check simple correct config processing
  File.open("conf.txt");
  if (File.is_open()) {
    File << "SYCL_PRINT_EXECUTION_GRAPH=before_addCG" << std::endl;
    File.close();
  }
  sycl::detail::readConfig(true);
  EXPECT_EQ(std::string("before_addCG"),
            sycl::detail::SYCLConfig<
                sycl::detail::SYCL_PRINT_EXECUTION_GRAPH>::get());

  // Check correct config processing
  File.open("conf.txt");
  if (File.is_open()) {
    File << "#a\r\n # b \r\n #a=b\r\n # a = "
            "b\r\n\r\nSYCL_PRINT_EXECUTION_GRAPH=after_addCG\r\n"
         << std::endl;
    File.close();
  }
  sycl::detail::readConfig(true);
  EXPECT_EQ(std::string("after_addCG"),
            sycl::detail::SYCLConfig<
                sycl::detail::SYCL_PRINT_EXECUTION_GRAPH>::get());

  // Check multi assignment
  File.open("conf.txt");
  if (File.is_open()) {
    File << "a=b\r\nb=c\nSYCL_PRINT_EXECUTION_GRAPH=before_addCopyBack\r\nc=d";
    File.close();
  }
  sycl::detail::readConfig(true);
  EXPECT_EQ(std::string("before_addCopyBack"),
            sycl::detail::SYCLConfig<
                sycl::detail::SYCL_PRINT_EXECUTION_GRAPH>::get());

  // Check assignment with comment
  File.open("conf.txt");
  if (File.is_open()) {
    File << "SYCL_PRINT_EXECUTION_GRAPH=after_addCopyBack #comment\r\n";
    File.close();
  }
  sycl::detail::readConfig(true);
  EXPECT_EQ(std::string("after_addCopyBack"),
            sycl::detail::SYCLConfig<
                sycl::detail::SYCL_PRINT_EXECUTION_GRAPH>::get());
}
