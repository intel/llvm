// RUN: %clang_cc1 -fsycl-is-device -internal-isystem -sycl-std=2020 -fsycl-int-header=%t.h %s -o %t.out
// RUN: FileCheck -input-file=%t.h %s
// RUN: %clang_cc1 -fsycl-is-host -sycl-std=2020 %s | FileCheck -input-file=%t.h %s

#include "Inputs/sycl.hpp"

// Check that meaningful information is returned when NDEBUG is not defined
// and empty strings and 0s are emitted when it is.
int test1() {
  cl::sycl::queue q;
  q.submit([&](cl::sycl::handler &h) { h.single_task([] {}); });
  q.submit([&](cl::sycl::handler &h) { h.single_task<class KernelName>([]() {}); });
  return 0;
}
// CHECK: template <> struct KernelInfoData<'_', 'Z', 'T', 'S', 'Z', 'Z', '5', 't', 'e', 's', 't', '1', 'v', 'E', 'N', 'K', 'U', 'l', 'R', 'N', '2', 'c', 'l', '4', 's', 'y', 'c', 'l', '7', 'h', 'a', 'n', 'd', 'l', 'e', 'r', 'E', 'E', '_', 'c', 'l', 'E', 'S', '2', '_', 'E', 'U', 'l', 'v', 'E', '_'> {
// CHECK:   static constexpr const char* getFileName() {
// CHECK: #ifndef NDEBUG
// CHECK:     return "code_location.cpp";
// CHECK: #else
// CHECK:     return "";
// CHECK: #endif
// CHECK:   }
// CHECK:   static constexpr const char* getFunctionName() {
// CHECK: #ifndef NDEBUG
// CHECK:     return "";
// CHECK: #else
// CHECK:     return "";
// CHECK: #endif
// CHECK:   }
// CHECK:   static constexpr unsigned getLineNumber() {
// CHECK: #ifndef NDEBUG
// CHECK:     return 11;
// CHECK: #else
// CHECK:     return 0;
// CHECK: #endif
// CHECK:   }
// CHECK:   static constexpr unsigned getColumnNumber() {
// CHECK: #ifndef NDEBUG
// CHECK:     return 54;
// CHECK: #else
// CHECK:     return 0;
// CHECK: #endif
// CHECK:   }
// CHECK: };

// CHECK: template <> struct KernelInfo<KernelName> {
// CHECK:   static constexpr const char* getFileName() {
// CHECK: #ifndef NDEBUG
// CHECK:     return "code_location.cpp";
// CHECK: #else
// CHECK:     return "";
// CHECK: #endif
// CHECK:   }
// CHECK:   __SYCL_DLL_LOCAL
// CHECK:   static constexpr const char* getFunctionName() {
// CHECK: #ifndef NDEBUG
// CHECK:     return "KernelName";
// CHECK: #else
// CHECK:     return "";
// CHECK: #endif
// CHECK:   }
// CHECK:   __SYCL_DLL_LOCAL
// CHECK:   static constexpr unsigned getLineNumber() {
// CHECK: #ifndef NDEBUG
// CHECK:     return 12;
// CHECK: #else
// CHECK:     return 0;
// CHECK: #endif
// CHECK:   }
// CHECK:   __SYCL_DLL_LOCAL
// CHECK:   static constexpr unsigned getColumnNumber() {
// CHECK: #ifndef NDEBUG
// CHECK:     return 72;
// CHECK: #else
// CHECK:     return 0;
// CHECK: #endif
// CHECK:   }
// CHECK: };

// Check that the right name and location is returned when
// lambda and kernel name are defined on different lines
class KernelName2;
int test2() {
  cl::sycl::queue q;
  q.submit([&](cl::sycl::handler &h) { h.single_task<KernelName2>(
                                           [] { int i = 2; }); });
  return 0;
}
// CHECK: template <> struct KernelInfo<::KernelName2> {
// CHECK:   static constexpr const char* getFileName() {
// CHECK: #ifndef NDEBUG
// CHECK:     return "code_location.cpp";
// CHECK: #else
// CHECK:     return "";
// CHECK: #endif
// CHECK:   }
// CHECK:   static constexpr const char* getFunctionName() {
// CHECK: #ifndef NDEBUG
// CHECK:     return "::KernelName2";
// CHECK: #else
// CHECK:     return "";
// CHECK: #endif
// CHECK:   }
// CHECK:   static constexpr unsigned getLineNumber() {
// CHECK: #ifndef NDEBUG
// CHECK:     return 86;
// CHECK: #else
// CHECK:     return 0;
// CHECK: #endif
// CHECK:   }
// CHECK:   static constexpr unsigned getColumnNumber() {
// CHECK: #ifndef NDEBUG
// CHECK:     return 44;
// CHECK: #else
// CHECK:     return 0;
// CHECK: #endif
// CHECK:   }
// CHECK: };

// Check that fully qualified name is returned
template <typename T> class KernelName3;
int test3() {
  cl::sycl::queue q;
  q.submit([&](cl::sycl::handler &h) { h.single_task<KernelName3<KernelName2>>(
                                           [] { int i = 3; }); });
  return 0;
}
// CHECK: template <> struct KernelInfo<::KernelName3<::KernelName2>> {
// CHECK:   static constexpr const char* getFileName() {
// CHECK: #ifndef NDEBUG
// CHECK:     return "code_location.cpp";
// CHECK: #else
// CHECK:     return "";
// CHECK: #endif
// CHECK:   }
// CHECK:   static constexpr const char* getFunctionName() {
// CHECK: #ifndef NDEBUG
// CHECK:     return "::KernelName3<::KernelName2>";
// CHECK: #else
// CHECK:     return "";
// CHECK: #endif
// CHECK:   }
// CHECK:   static constexpr unsigned getLineNumber() {
// CHECK: #ifndef NDEBUG
// CHECK:     return 125;
// CHECK: #else
// CHECK:     return 0;
// CHECK: #endif
// CHECK:   }
// CHECK:   static constexpr unsigned getColumnNumber() {
// CHECK: #ifndef NDEBUG
// CHECK:     return 44;
// CHECK: #else
// CHECK:     return 0;
// CHECK: #endif
// CHECK:   }
// CHECK: };

// Check that the location information returned is that of l4
auto l4 = []() { return 4; };
int test4() {
  cl::sycl::queue q;
  q.submit([=](cl::sycl::handler &h) { h.single_task<class KernelName4>(l4); });
  return 0;
}
// CHECK: template <> struct KernelInfo<KernelName4> {
// CHECK:   static constexpr const char* getFileName() {
// CHECK: #ifndef NDEBUG
// CHECK:     return "code_location.cpp";
// CHECK: #else
// CHECK:     return "";
// CHECK: #endif
// CHECK:   }
// CHECK:   static constexpr const char* getFunctionName() {
// CHECK: #ifndef NDEBUG
// CHECK:     return "KernelName4";
// CHECK: #else
// CHECK:     return "";
// CHECK: #endif
// CHECK:   }
// CHECK:   static constexpr unsigned getLineNumber() {
// CHECK: #ifndef NDEBUG
// CHECK:     return 160;
// CHECK: #else
// CHECK:     return 0;
// CHECK: #endif
// CHECK:   }
// CHECK:   static constexpr unsigned getColumnNumber() {
// CHECK: #ifndef NDEBUG
// CHECK:     return 11;
// CHECK: #else
// CHECK:     return 0;
// CHECK: #endif
// CHECK:   }
// CHECK: };

// Check that fully qualified name is returned when unnamed lambda
// kernel is enclosed in a namespace
namespace NS {
int test5() {
  cl::sycl::queue q;
  q.submit([=](cl::sycl::handler &h) { h.single_task([] {}); });
  q.submit([=](cl::sycl::handler &h) { h.single_task<class KernelName5>([] {}); });
  return 0;
}
} // namespace NS
// CHECK: template <> struct KernelInfoData<'_', 'Z', 'T', 'S', 'Z', 'Z', 'N', '2', 'N', 'S', '5', 't', 'e', 's', 't', '5', 'E', 'v', 'E', 'N', 'K', 'U', 'l', 'R', 'N', '2', 'c', 'l', '4', 's', 'y', 'c', 'l', '7', 'h', 'a', 'n', 'd', 'l', 'e', 'r', 'E', 'E', '_', 'c', 'l', 'E', 'S', '3', '_', 'E', 'U', 'l', 'v', 'E', '_'> {
// CHECK:   static constexpr const char* getFileName() {
// CHECK: #ifndef NDEBUG
// CHECK:     return "code_location.cpp";
// CHECK: #else
// CHECK:     return "";
// CHECK: #endif
// CHECK:   }
// CHECK:   static constexpr const char* getFunctionName() {
// CHECK: #ifndef NDEBUG
// CHECK:     return "NS::";
// CHECK: #else
// CHECK:     return "";
// CHECK: #endif
// CHECK:   }
// CHECK:   static constexpr unsigned getLineNumber() {
// CHECK: #ifndef NDEBUG
// CHECK:     return 202;
// CHECK: #else
// CHECK:     return 0;
// CHECK: #endif
// CHECK:   }
// CHECK:   static constexpr unsigned getColumnNumber() {
// CHECK: #ifndef NDEBUG
// CHECK:     return 54;
// CHECK: #else
// CHECK:     return 0;
// CHECK: #endif
// CHECK:   }
// CHECK: };
// CHECK: template <> struct KernelInfo<NS::KernelName5> {
// CHECK:   static constexpr const char* getFileName() {
// CHECK: #ifndef NDEBUG
// CHECK:     return "code_location.cpp";
// CHECK: #else
// CHECK:     return "";
// CHECK: #endif
// CHECK:   }
// CHECK:   static constexpr const char* getFunctionName() {
// CHECK: #ifndef NDEBUG
// CHECK:     return "NS::KernelName5";
// CHECK: #else
// CHECK:     return "";
// CHECK: #endif
// CHECK:   }
// CHECK:   static constexpr unsigned getLineNumber() {
// CHECK: #ifndef NDEBUG
// CHECK:     return 203;
// CHECK: #else
// CHECK:     return 0;
// CHECK: #endif
// CHECK:   }
// CHECK:   static constexpr unsigned getColumnNumber() {
// CHECK: #ifndef NDEBUG
// CHECK:     return 73;
// CHECK: #else
// CHECK:     return 0;
// CHECK: #endif
// CHECK:   }
// CHECK: };

// Check that the location information returned is that of the Functor
struct Functor {
  void operator()() const {
  }
};
int test6() {
  Functor F;
  cl::sycl::queue q;
  q.submit([=](cl::sycl::handler &h) { h.single_task<class KernelName6>(F); });
  return 0;
}
// CHECK: template <> struct KernelInfo<KernelName6> {
// CHECK:   static constexpr const char* getFileName() {
// CHECK: #ifndef NDEBUG
// CHECK:     return "code_location.cpp";
// CHECK: #else
// CHECK:     return "";
// CHECK: #endif
// CHECK:   }
// CHECK:   static constexpr const char* getFunctionName() {
// CHECK: #ifndef NDEBUG
// CHECK:     return "KernelName6";
// CHECK: #else
// CHECK:     return "";
// CHECK: #endif
// CHECK:   }
// CHECK:   static constexpr unsigned getLineNumber() {
// CHECK: #ifndef NDEBUG
// CHECK:     return 269;
// CHECK: #else
// CHECK:     return 0;
// CHECK: #endif
// CHECK:   }
// CHECK:   static constexpr unsigned getColumnNumber() {
// CHECK: #ifndef NDEBUG
// CHECK:     return 8;
// CHECK: #else
// CHECK:     return 0;
// CHECK: #endif
// CHECK:   }
// CHECK: };
