// RUN: %clangxx -fsycl -DNDEBUG %s -o %t.out
// RUN: %t.out
// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %t.out

#include <cassert>
#include <iostream>
#include <sycl/sycl.hpp>

int main() {
  auto code_loc = sycl::detail::code_location::current();
  const char *funcName = "main";
#ifdef NDEBUG
  if (code_loc.fileName() != nullptr)
    return 1;
  if (code_loc.functionName() != funcName)
    return 1;
  if (code_loc.lineNumber() != 11)
    return 1;
  if (code_loc.columnNumber() != 19)
    return 1;
#else
  assert((code_loc.fileName() != nullptr));
  std::string str = code_loc.fileName();
  assert(((str.find("code_location.cpp") != std::string::npos) &&
          "Filename is wrong"));
  assert((code_loc.functionName() != nullptr));
  str = code_loc.functionName();
  assert(
      ((str.find(funcName) != std::string::npos) && "Function name is wrong"));
  assert((code_loc.lineNumber() != 0));
  assert(((code_loc.lineNumber() == 11) && "Line number is wrong"));
  assert((code_loc.columnNumber() != 0));
  assert(((code_loc.columnNumber() == 19) && "Column number is wrong"));
#endif
  return 0;
}
