// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %RUN_ON_HOST %t.out

#include <CL/sycl.hpp>

using namespace cl::sycl;

int main() {
  // Test new constructors, initially each with empty string messages.
  std::string emptyStr;
  const char *emptyCharPtr = "";

  // Going to set the error code values to each of the enum (0-12).
  exception ex1(make_error_code(errc::runtime), emptyStr);
  exception ex2(make_error_code(errc::kernel), emptyCharPtr);
  exception ex3(make_error_code(errc::accessor));
  exception ex4(static_cast<int>(errc::nd_range), sycl_category(), emptyStr);
  exception ex5(static_cast<int>(errc::event), sycl_category(), emptyCharPtr);
  exception ex6(static_cast<int>(errc::kernel_argument), sycl_category());

  queue Q;
  context ctx = Q.get_context();
  exception ex7(ctx, make_error_code(errc::build), emptyStr);
  exception ex8(ctx, make_error_code(errc::invalid), emptyCharPtr);
  exception ex9(ctx, make_error_code(errc::memory_allocation));
  exception ex10(ctx, static_cast<int>(errc::platform), sycl_category(),
                 emptyStr);
  exception ex11(ctx, static_cast<int>(errc::profiling), sycl_category(),
                 emptyCharPtr);
  exception ex12(ctx, static_cast<int>(errc::feature_not_supported),
                 sycl_category());

  std::vector<exception> v{ex1, ex2, ex3, ex4,  ex5,  ex6,
                           ex7, ex8, ex9, ex10, ex11, ex12};
  for (int i = 0; i < 12; i++) {
    exception ex = v[i];
    assert(ex.code().value() == i && "unexpected error_code.value() retrieved");
    assert(ex.category() == sycl_category() && "expected SYCL error category");
    if (i < 6) {
      assert(!ex.has_context() &&
             "none of the first six exceptions should have a context");
    } else {
      assert(ex.has_context() && ex.get_context() == ctx &&
             "the second six exceptions should have a context equal to ctx");
    }
    assert(strlen(ex.what()) == 0 &&
           "all these exceptions were initialized with empty strings. We "
           "should not have anything in the 'what' message");
  }

  // Now test constructor with a real string value.
  std::string testString("this is a test");
  exception ex_string1(make_error_code(errc::kernel_not_supported), testString);
  assert(testString.compare(ex_string1.what()) == 0);
  char testCharPtr[] = "this is also a test";
  exception ex_string2(make_error_code(errc::backend_mismatch), testCharPtr);
  assert(strcmp(ex_string2.what(), testCharPtr) == 0);

  // Test sycl_category.
  assert(std::string("SYCL").compare(sycl_category().name()) == 0 &&
         "sycl_category name should be SYCL");

  // Test make_error_code.
  std::error_code ec = make_error_code(errc::feature_not_supported);
  assert(ec.value() == static_cast<int>(errc::feature_not_supported));
  assert(std::string("SYCL").compare(ec.category().name()) == 0 &&
         "error code category name should be SYCL");

  std::cout << "OK" << std::endl;
  return 0;
}