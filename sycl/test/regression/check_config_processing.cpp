// RUN: %clangxx -fsycl %s -o %t.out
//
// RUN: env TEST=SPACE_AT_FIRST_POSITION %t.out > %t1.conf
// RUN: env TEST=EXCESS_SPACE env SYCL_CONFIG_FILE_NAME=%t1.conf %t.out
//
// RUN: env TEST=SPACE_AT_LAST_POSITION %t.out > %t2.conf
// RUN: env TEST=EXCESS_SPACE env SYCL_CONFIG_FILE_NAME=%t2.conf %t.out
//
// RUN: env TEST=SPACE_BEFORE_ASSIGNMENT %t.out > %t3.conf
// RUN: env TEST=EXCESS_SPACE env SYCL_CONFIG_FILE_NAME=%t3.conf %t.out
//
// RUN: env TEST=SPACE_AFTER_ASSIGNMENT %t.out > %t4.conf
// RUN: env TEST=EXCESS_SPACE env SYCL_CONFIG_FILE_NAME=%t4.conf %t.out
//
// RUN: env TEST=VARIABLE_NAME_BIGGER_THAN_MAX_CONFIG_NAME %t.out > %t5.conf
// RUN: env TEST=INCORRECT_VARIABLE_NAME env SYCL_CONFIG_FILE_NAME=%t5.conf %t.out
//
// RUN: env TEST=VARIABLE_WITHOUT_NAME %t.out > %t6.conf
// RUN: env TEST=INCORRECT_VARIABLE_NAME env SYCL_CONFIG_FILE_NAME=%t6.conf %t.out
//
// RUN: env TEST=VARIABLE_VALUE_BIGGER_THAN_MAX_CONFIG_VALUE %t.out > %t7.conf
// RUN: env TEST=INCORRECT_VARIABLE_VALUE env SYCL_CONFIG_FILE_NAME=%t7.conf %t.out
//
// RUN: env TEST=VARIABLE_WITHOUT_VALUE %t.out > %t8.conf
// RUN: env TEST=INCORRECT_VARIABLE_VALUE env SYCL_CONFIG_FILE_NAME=%t8.conf %t.out
//
// RUN: env TEST=COMPLEX_CONFIG %t.out > %t9.conf
// RUN: env TEST=INCORRECT_VARIABLE_VALUE env SYCL_CONFIG_FILE_NAME=%t9.conf %t.out
//
// RUN: env TEST=SET_CORRECT_ENVIRONMENT %t.out > %t10.conf
// RUN: env TEST=CORRECT_CONFIG_FILE env SYCL_CONFIG_FILE_NAME=%t10.conf %t.out
// RUN: ls | grep dot
// RUN: rm *.dot

#include <CL/sycl.hpp>
#include <iostream>
#include <regex>

int main() {
  std::string testEnvVarValue;
  if (getenv("TEST")) {
    testEnvVarValue = getenv("TEST");
  } else {
    throw sycl::exception(sycl::make_error_code(sycl::errc::runtime),
                          "Environment is not found");
  }

  // Сonfig creation
  if (testEnvVarValue == "SPACE_AT_FIRST_POSITION") {
    std::cout << " a=b" << std::endl;
    return 0;
  }
  if (testEnvVarValue == "SPACE_AT_LAST_POSITION") {
    std::cout << "a=b " << std::endl;
    return 0;
  }
  if (testEnvVarValue == "SPACE_BEFORE_ASSIGNMENT") {
    std::cout << "a =b" << std::endl;
    return 0;
  }
  if (testEnvVarValue == "SPACE_AFTER_ASSIGNMENT") {
    std::cout << "a= b" << std::endl;
    return 0;
  }
  if (testEnvVarValue == "VARIABLE_NAME_BIGGER_THAN_MAX_CONFIG_NAME") {
    // Max variable name is 256 characters
    for (int i = 0; i <= 256; i++)
      std::cout << "a";
    std::cout << "=b" << std::endl;
    return 0;
  }
  if (testEnvVarValue == "VARIABLE_WITHOUT_NAME") {
    std::cout << "=b" << std::endl;
    return 0;
  }
  if (testEnvVarValue == "VARIABLE_VALUE_BIGGER_THAN_MAX_CONFIG_VALUE") {
    // Max variable value contains 256 or 1024 characters
    std::cout << "a=";
    for (int i = 0; i <= 1024; i++)
      std::cout << "b";
    std::cout << std::endl;
    return 0;
  }
  if (testEnvVarValue == "VARIABLE_WITHOUT_VALUE") {
    std::cout << "a=" << std::endl;
    return 0;
  }
  if (testEnvVarValue == "COMPLEX_CONFIG") {
    // To check that the processing reaches "a="
    std::cout << "#\n"
              << "  #\n"
              << "#a=b"
              << "\n\n\n"
              << "a\n"
              << " aaa \n"
              << "a=b\n"
              << "a=\n"
              << std::endl;
    return 0;
  }
  if (testEnvVarValue == "SET_CORRECT_ENVIRONMENT") {
    std::cout << "#\n"
              << "#abc\n"
              << "# abc = cba \r\n"
              << "\r\n"
              << "SYCL_PRINT_EXECUTION_GRAPH=always" << std::endl;
    return 0;
  }

  // Config check
  if (testEnvVarValue == "EXCESS_SPACE") {
    try {
      auto device = sycl::device{sycl::default_selector{}};
    } catch (sycl::exception &e) {
      std::string errorMessage = e.what();
      if (errorMessage ==
          "SPACE found at the beginning/end of the line or before/after '='") {
        return 0;
      } else {
        throw e;
      }
    }
  }
  if (testEnvVarValue == "INCORRECT_VARIABLE_NAME") {
    try {
      auto device = sycl::device{sycl::default_selector{}};
    } catch (sycl::exception &e) {
      std::string errorMessage = e.what();
      std::regex regular(
          "Variable name is more than ([\\d]+) or less than one character");
      if (std::regex_match(errorMessage, regular)) {
        return 0;
      } else {
        throw e;
      }
    }
  }
  if (testEnvVarValue == "INCORRECT_VARIABLE_VALUE") {
    try {
      auto device = sycl::device{sycl::default_selector{}};
    } catch (sycl::exception &e) {
      std::string errorMessage = e.what();
      std::regex regular("The value contains more than ([\\d]+) characters or "
                         "does not contain them at all");
      if (std::regex_match(errorMessage, regular)) {
        return 0;
      } else {
        throw e;
      }
    }
  }
  if (testEnvVarValue == "CORRECT_CONFIG_FILE") {
    sycl::buffer<int, 1> Buf(sycl::range<1>{1});
    auto Acc = Buf.get_access<sycl::access::mode::read>();
    return 0;
  }
  throw std::logic_error("Environment is incorrect");
}
