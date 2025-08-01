// UNSUPPORTED: cuda
// UNSUPPORTED-TRACKER: https://github.com/intel/llvm/issues/19214
// RUN: %{build} -DNDEBUG -o %t1.out
// RUN: %{run} %t1.out | FileCheck %s

// RUN: %{build} -o %t2.out
// RUN: %{run} %t2.out | FileCheck %s

/*
    clang++ -fsycl -DNDEBUG -o smyl.bin code_location_e2e.cpp   // <<--- NDEBUG
   will suppress fileLocation

    -DNDEBUG  <-- not debugging, meaning fileName should not be retrievable (for
   data privacy reasons)

   Between having to test against NDEBUG and test from kernel code,
   using assert/abort type of logic is not available. This test just
   outputs strings that it checks with FileCheck.

   This test uses hardcoded file code location positions. If modified, those
   may change. pass in any value to signal "report only" and not run tests.

*/

#include <cstring>
#include <sycl/detail/core.hpp>

#include <sycl/stream.hpp>
using namespace sycl;

template <typename OS> void report(OS &out, detail::code_location code_loc) {
  out << "function {line:col} => " << code_loc.functionName() << " {"
      << code_loc.lineNumber() << ":" << code_loc.columnNumber() << "}"
      << "\n";

  auto fileName = code_loc.fileName();
  if (fileName == nullptr)
    out << "nullptr for fileName"
        << "\n";
  else
    out << "fileName: " << code_loc.fileName() << "\n";
}

template <typename OS>
void test(OS &out, detail::code_location &code_loc, const char *fileName,
          const char *funcName, int line, int col) {

  // functionName
  auto funcNameStr = code_loc.functionName();
  auto fNameResult =
      ((funcNameStr != nullptr) && !strcmp(funcNameStr, funcName)) ? "OK"
                                                                   : "WRONG";
  out << "code_location.functionName: " << fNameResult << "\n";

  // lineNumber
  auto lineNumberResult = (code_loc.lineNumber() == line) ? "OK" : "WRONG";
  out << "code_location.lineNumber: " << lineNumberResult << "\n";

  // columnNumber
  auto colNumberResult = (code_loc.columnNumber() == col) ? "OK" : "WRONG";
  out << "code_location.columnNumber: " << colNumberResult << "\n";

  // fileName
  auto fileNameStr = code_loc.fileName();
#ifdef NDEBUG
  // NDEBUG == not debugging == no fileName (for security).
  auto fileNameResult =
      (fileNameStr == nullptr)
          ? "OK"
          : "WRONG - fileName should not be present when NDEBUG defined";
#else
  auto fileNameResult = (strcmp(fileName, fileNameStr) == 0) ? "OK" : "WRONG";
#endif
  out << "code_location.fileName: " << fileNameResult << "\n";
}

int main(int argc, char **argv) {
  bool testing =
      (argc <= 1); // Passing in ANY argument means report only, do not test.
  auto code_loc = sycl::detail::code_location::current(); // <--
  int EXPECTED_LINE = __LINE__ - 1;
#ifdef NDEBUG
  std::cout << "NDEBUG, therefore no fileName" << std::endl;
#else
  std::cout << "NDEBUG NOT DEFINED, make sure file name isn't a full path "
            << std::endl;
  std::cout << "file name: " << code_loc.fileName() << std::endl;
#endif
  std::cout << "------- host test -------" << std::endl;
  report(std::cout, code_loc);

  if (testing)
    test(std::cout, code_loc, "code_location_e2e.cpp", "main", EXPECTED_LINE,
         19);

  std::cout << "------- kernel test -------" << std::endl;

  queue q;
  q.submit([testing](handler &cgh) {
    sycl::stream out(2024, 400, cgh);
    cgh.single_task<class KHALID>([=]() {
      auto kernel_loc = sycl::detail::code_location::current(); // <--
      int EXPECTED_LINE = __LINE__ - 1;
      report(out, kernel_loc);

      if (testing)
        test(out, kernel_loc, "code_location_e2e.cpp", "operator()",
             EXPECTED_LINE, 25);

      out << "-------" << sycl::endl;
    });
  });
  q.wait();

  return 0;
}

// CHECK: ------- host test -------
// CHECK:      code_location.functionName: OK
// CHECK-NEXT: code_location.lineNumber: OK
// CHECK-NEXT: code_location.columnNumber: OK
// XHECK-NEXT: code_location.fileName: OK

// CHECK: ------- kernel test -------
// CHECK:      code_location.functionName: OK
// CHECK-NEXT: code_location.lineNumber: OK
// CHECK-NEXT: code_location.columnNumber: OK
// XHECK-NEXT: code_location.fileName: OK
