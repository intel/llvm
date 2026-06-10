//==---------- code_location.hpp ----- code_location utilities ------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/defines_elementary.hpp> // for __has_builtin fallback
#include <sycl/detail/export.hpp>             // for __SYCL_EXPORT

#include <cstdint>

namespace sycl {
inline namespace _V1 {
namespace detail {

#if !defined(NDEBUG) && (_MSC_VER > 1929 || __has_builtin(__builtin_FILE))
#define __CODELOC_FILE_NAME __builtin_FILE()
#else
#define __CODELOC_FILE_NAME nullptr
#endif

#if _MSC_VER > 1929 || __has_builtin(__builtin_FUNCTION)
#define __CODELOC_FUNCTION __builtin_FUNCTION()
#else
#define __CODELOC_FUNCTION nullptr
#endif

#if _MSC_VER > 1929 || __has_builtin(__builtin_LINE)
#define __CODELOC_LINE __builtin_LINE()
#else
#define __CODELOC_LINE 0
#endif

#if _MSC_VER > 1929 || __has_builtin(__builtin_COLUMN)
#define __CODELOC_COLUMN __builtin_COLUMN()
#else
#define __CODELOC_COLUMN 0
#endif

// Data structure that captures the user code location information using the
// builtin capabilities of the compiler
struct code_location {
  static constexpr code_location
  current(const char *fileName = __CODELOC_FILE_NAME,
          const char *funcName = __CODELOC_FUNCTION,
          uint32_t lineNo = __CODELOC_LINE,
          uint32_t columnNo = __CODELOC_COLUMN) noexcept {
    return code_location(fileName, funcName, lineNo, columnNo);
  }

#undef __CODELOC_FILE_NAME
#undef __CODELOC_FUNCTION
#undef __CODELOC_LINE
#undef __CODELOC_COLUMN

  constexpr code_location(const char *file, const char *func, uint32_t line,
                          uint32_t col) noexcept
      : MFileName(file), MFunctionName(func), MLineNo(line), MColumnNo(col) {}

  constexpr code_location() noexcept
      : MFileName(nullptr), MFunctionName(nullptr), MLineNo(0u), MColumnNo(0u) {
  }

  constexpr uint32_t lineNumber() const noexcept {
    return static_cast<uint32_t>(MLineNo);
  }
  constexpr uint32_t columnNumber() const noexcept {
    return static_cast<uint32_t>(MColumnNo);
  }
  constexpr const char *fileName() const noexcept { return MFileName; }
  constexpr const char *functionName() const noexcept { return MFunctionName; }

private:
  const char *MFileName;
  const char *MFunctionName;
  uint32_t MLineNo;
  uint32_t MColumnNo;
};

/// @brief Data type that manages the code_location information in TLS
/// @details As new SYCL features are added, they all enable the propagation of
/// the code location information where the SYCL API was called by the
/// application layer. In order to facilitate this, the tls_code_loc_t object
/// assists in managing the data in TLS :
///   (1) Populate the information when you at the top level function in the
///   call chain. This is usually the end-user entry point function into SYCL.
///   (2) Remove the information when the object goes out of scope in the top
///   level function.
///
/// Usage:-
///   void bar() {
///     tls_code_loc_t p;
///     // Print the source information of where foo() was called in main()
///     std::cout << p.query().fileName() << ":" << p.query().lineNumber() <<
///     std::endl;
///   }
///   // Will work for arbitrary call chain lengths.
///   void bar1() {bar();}
///
///   // Foo() is equivalent to a SYCL end user entry point such as
///   // queue.memcpy() or queue.copy()
///   void foo(const code_location &loc) {
///     tls_code_loc_t tp(loc);
///     bar1();
///   }
///
///   void main() {
///     foo(const code_location &loc = code_location::current());
///   }
class __SYCL_EXPORT tls_code_loc_t {
public:
  /// @brief Consructor that checks to see if a TLS entry already exists
  /// @details If a previous populated TLS entry exists, this constructor will
  /// capture the informationa and allow you to query the information later.
  tls_code_loc_t();
  /// @brief Iniitializes TLS with CodeLoc if a TLS entry not present
  /// @param CodeLoc The code location information to set up the TLS slot with.
  tls_code_loc_t(const detail::code_location &CodeLoc);

  // Used to maintain global state (GCodeLocTLS), so we do not want to copy
  tls_code_loc_t(const tls_code_loc_t &) = delete;
  tls_code_loc_t &operator=(const tls_code_loc_t &) = delete;

  /// If the code location is set up by this instance, reset it.
  ~tls_code_loc_t();
  /// @brief  Query the information in the TLS slot
  /// @return The code location information saved in the TLS slot. If not TLS
  /// entry has been set up, a default coe location is returned.
  const detail::code_location &query();
  /// @brief Returns true if the TLS slot was cleared when this object was
  /// constructed.
  bool isToplevel() const { return !MLocalScope; }

private:
  // Cache the TLS location to decrease amount of TLS accesses.
  detail::code_location &CodeLocTLSRef;
  // The flag that is used to determine if the object is in a local scope or in
  // the top level scope.
  bool MLocalScope = true;
};

} // namespace detail
} // namespace _V1
} // namespace sycl
