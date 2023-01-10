// RUN: %clangxx -fsycl %s -o %t.out
//
// RUN: env TEST=SET_CORRECT_ENVIRONMENT %t.out > %t.conf
// RUN: env TEST=CORRECT_CONFIG_FILE env SYCL_CONFIG_FILE_NAME=%t.conf %t.out
// RUN: ls *.dot
// RUN: rm *.dot

#include <iostream>
#include <regex>
#include <sycl/sycl.hpp>

int main() {
  std::string testEnvVarValue = getenv("TEST");

  // Сonfig creation
  if (testEnvVarValue == "SET_CORRECT_ENVIRONMENT") {
    std::cout << "#\n"
              << "#abc\n"
              << "# abc = cba \r\n"
              << "\r\n"
              << "SYCL_PRINT_EXECUTION_GRAPH=always\r\n"
              << std::endl;
    return 0;
  }

  // Config check
  if (testEnvVarValue == "CORRECT_CONFIG_FILE") {
    sycl::buffer<int, 1> Buf(sycl::range<1>{1});
    auto Acc = Buf.get_access<sycl::access::mode::read>();
    return 0;
  }
  throw std::logic_error("Environment is incorrect");
}
