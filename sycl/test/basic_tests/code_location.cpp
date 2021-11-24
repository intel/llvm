// RUN: %clangxx -fsycl -DNDEBUG %s -o %t.out
// RUN: %t.out
// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %t.out

#include <CL/sycl.hpp>
#include <cassert>
#include <iostream>

int main() {
  auto code_loc = sycl::detail::code_location::current();
  const char *currentFunction = "main";
#ifdef NDEBUG
  if (code_loc.fileName() != nullptr) {
    return 1;
  }
  if (code_loc.functionName() != currentFunction) {
    return 1;
  }
  if (code_loc.lineNumber() != 11) {
    return 1;
  }
  if (code_loc.columnNumber() != 21) {
    return 1;
  }
// std::cout << "NDEBUG asserts passed" << std::endl;
#else
  assert((code_loc.fileName() != nullptr));
  std::string str = code_loc.fileName();
  assert(((str.find("code_location.cpp") != std::string::npos) &&
          "Filename is wrong"));
  assert((code_loc.functionName() != nullptr));
  str = code_loc.functionName();
  assert(((str.find(currentFunction) != std::string::npos) &&
          "Function name is wrong"));
  assert((code_loc.lineNumber() != 0));
  assert(((code_loc.lineNumber() == 11) && "Line number is wrong"));
  assert((code_loc.columnNumber() != 0));
  assert(((code_loc.columnNumber() == 21) && "Column number is wrong"));
// std::cout << "asserts without NDEBUG passed" << std::endl;
#endif
  return 0;
}