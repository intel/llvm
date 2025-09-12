// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %t.out
#include <sycl/sycl.hpp>

int main() {
  // Default constructor
  sycl::detail::string empty_s;
  assert(empty_s.empty() && "Default constructed string should be empty");
  assert(strcmp(empty_s.c_str(), "") == 0 &&
         "Default constructed string should be empty string");

  // std::string_view constructor
  std::string_view sv = "Hello, World!";
  sycl::detail::string s1(sv);
  assert(strcmp(s1.c_str(), "Hello, World!") == 0 &&
         "String from string_view constructor should match view content");

  // Copy constructor
  sycl::detail::string s2(s1);
  assert(strcmp(s2.c_str(), "Hello, World!") == 0 &&
         "Copied string should match original");
  // Check for deep copy
  s1 = "Changed";
  assert(strcmp(s2.c_str(), "Hello, World!") == 0 &&
         "Copied string should be a deep copy");

  // Move constructor
  sycl::detail::string s3(std::move(s1));
  assert(strcmp(s3.c_str(), "Changed") == 0 &&
         "Moved string should have original content");

  // string_view assignment
  sycl::detail::string s4;
  s4 = "New String";
  assert(strcmp(s4.c_str(), "New String") == 0 &&
         "String_view assignment should update content");

  // Copy assignment
  sycl::detail::string s5;
  s5 = s2;
  assert(strcmp(s5.c_str(), "Hello, World!") == 0 &&
         "Copy assignment should work correctly");

  // Move assignment
  sycl::detail::string s6;
  s6 = std::move(s3);
  assert(strcmp(s6.c_str(), "Changed") == 0 &&
         "Move assignment should work correctly");

  // Test c_str() and data()
  assert(strcmp(s6.data(), "Changed") == 0 &&
         "data() should return same as c_str()");

  // Test empty(). Thrice.
  sycl::detail::string s_not_empty("not empty");
  sycl::detail::string s_empty;
  assert(!s_not_empty.empty() && "String with content should not be empty");
  assert(sycl::detail::string("").empty() && "Empty string should be empty");
  assert(s_empty.empty() && "Default constructed string should be empty");

  // Test swap.
  sycl::detail::string swap_s1("first");
  sycl::detail::string swap_s2("second");
  swap(swap_s1, swap_s2);
  assert(strcmp(swap_s1.c_str(), "second") == 0 &&
         "swap should exchange content");
  assert(strcmp(swap_s2.c_str(), "first") == 0 &&
         "swap should exchange content");

  // Comparison operators
  sycl::detail::string compare_s("match");
  assert((compare_s == "match") && "string == string_view should work");
  assert(("match" == compare_s) && "string_view == string should work");
  assert(!(compare_s == "no match") && "string == string_view should fail");

  return 0;
}