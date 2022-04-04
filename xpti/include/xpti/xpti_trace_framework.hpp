//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
#pragma once

#include <atomic>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <memory>
#include <sstream>
#include <thread>

#include "xpti/xpti_data_types.h"
#include "xpti/xpti_trace_framework.h"
#include "xpti_trace_framework.h"

#if defined(_WIN32) || defined(_WIN64)
#include <string>
#include <strsafe.h>
// Windows.h defines min and max macros, that interfere with C++ std::min and
// std::max. The following definition disables that feature.
#define NOMINMAX
#include <windows.h>
#undef NOMINMAX
typedef HINSTANCE xpti_plugin_handle_t;
typedef FARPROC xpti_plugin_function_t;
#define XPTI_PLUGIN_STRING "*.dll"
#define XPTI_PATH_SEPARATOR "\\"
// Windows does not have PATH_MAX defined or is inconsistent; Documentation
// suggests that 32767 is the max string length of environment variables on
// Windows
constexpr auto WIN_PATH_MAX = 32767;
#else // Linux and MacOSX
#include <cstdlib>
#include <dlfcn.h>
#include <limits.h>
#include <string>

typedef void *xpti_plugin_handle_t;
typedef void *xpti_plugin_function_t;
#define XPTI_PATH_SEPARATOR "/"

#if defined(__unix__) // Linux
#define XPTI_PLUGIN_STRING "*.so"
#elif defined(__APPLE__) // Mac
#define XPTI_PLUGIN_STRING "*.dylib"
#endif
#endif

/// Insert something when compiled with msvc
/// https://docs.microsoft.com/en-us/cpp/preprocessor/predefined-macros
#ifdef _MSC_VER
#define __XPTI_INSERT_IF_MSVC(x) x
#else
#define __XPTI_INSERT_IF_MSVC(x)
#endif

namespace xpti {
namespace utils {

class StringHelper {
public:
  template <class T> std::string addressAsString(T address) {
    std::stringstream ss;
    ss << std::hex << address;
    return ss.str();
  }

  template <class T>
  std::string nameWithAddress(const char *prefix, T address) {
    std::string coded_string;

    if (prefix)
      coded_string = prefix;
    else
      coded_string = "unknown";

    coded_string += "[" + addressAsString<T>(address) + "]";
    return coded_string;
  }

  template <class T>
  std::string nameWithAddress(std::string &prefix, T address) {
    std::string coded_string;
    if (!prefix.empty())
      coded_string = prefix + "[" + addressAsString<T>(address) + "]";
    else
      coded_string = "unknown[" + addressAsString<T>(address) + "]";

    return coded_string;
  }

  std::string nameWithAddressString(const char *prefix, std::string &address) {
    std::string coded_string;

    if (prefix)
      coded_string = prefix;
    else
      coded_string = "unknown";

    coded_string += "[" + address + "]";
    return coded_string;
  }

  std::string nameWithAddressString(const std::string &prefix,
                                    std::string &address) {
    std::string coded_string;
    ;
    if (!prefix.empty())
      coded_string = prefix + "[" + address + "]";
    else
      coded_string = "unknown[" + address + "]";

    return coded_string;
  }
};

class PlatformHelper {
public:
  /// @brief Retrieves the last error and represents it as a std::string
  /// @details This function is a platform independent abstraction for
  /// retrieving the last error that was captured.
  ///
  /// @return <addr>        The last error logged in the system
  ///
  std::string getLastError() {
    std::string error;
#if defined(_WIN32) || defined(_WIN64)
    DWORD err = GetLastError();
    LPVOID msgBuff;
    size_t size = FormatMessageA(
        FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
            FORMAT_MESSAGE_IGNORE_INSERTS,
        NULL, err, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), (LPSTR)&msgBuff,
        0, NULL);

    if (size) {
      LPCSTR msgStr = (LPCSTR)msgBuff;
      error = std::string(msgStr, msgStr + size);
    }
    LocalFree(msgBuff);
#else
    const char *err_string = dlerror();
    if (err_string)
      error = err_string;
#endif
    return error;
  }

  std::string getEnvironmentVariable(const std::string &var) {
    // Previous code that used a secure API for getting the environment variable
    // and was replaced with a C++11 std::getenv for simplicity. However, if
    // environments require the use of the suggested API from Microsoft, the
    // code segment below may be un-commented and current std::getenv() code
    // block encapsulated in the #else clause.
    //
    // #if defined(_WIN32) || defined(_WIN64)
    //     // Implementation that uses the secure versions of the API to get an
    //     // environment variable.
    //     char *valuePtr = nullptr;
    //     size_t length;
    //     errno_t error = _dupenv_s(&valuePtr, &length, var.c_str());
    //     // Variable doesn't exist
    //     if (error)
    //       return "";
    //     // If the variable exists, then get the value into a temporary copy
    //     std::string retValue = valuePtr;
    //     free(valuePtr);
    //     return retValue;
    // #else

    // Remove irrelevant deprecation warning in this context in case /sdl option
    // is used to provide more information on deprecated API
    // https://docs.microsoft.com/en-us/cpp/c-runtime-library/security-features-in-the-crt
    // https://docs.microsoft.com/en-us/cpp/build/reference/sdl-enable-additional-security-checks
    __XPTI_INSERT_IF_MSVC(__pragma(warning(suppress : 4996)))
    const char *val = std::getenv(var.c_str());
    return val ? val : "";
    // #endif
  }

  /// @brief Finds a function defined by symbol in a shared object or DLL
  /// @details This function is a platform independent abstraction for finding
  /// a symbol in a shared object of DLL. If successful, it will return a
  /// valid function pointer.
  ///
  /// @param [in]  h        The plugin handle which is required to make this
  ///                       query
  /// @param [in] symbol    The symbol that needs to be searched within the
  ///                       shared object or DLL represented by the handle h
  ///
  /// @return <addr>        If the query is successful, the function returns
  ///                       a valid address
  /// @return nullptr       If the query fails, nullptr is returned
  ///
  xpti_plugin_function_t findFunction(xpti_plugin_handle_t h,
                                      const char *symbol) {
    xpti_plugin_function_t func = nullptr;
    if (h && symbol) {
#if defined(_WIN32) || defined(_WIN64)
      func = GetProcAddress(h, symbol);
#else
      func = dlsym(h, symbol);
#endif
    }
    return func;
  }

  /// @brief Loads a shared object or DLL and returns a plugin handle
  /// @details This function is a platform independent abstraction for loading
  /// a DLL or shared object and returns a valid plugin handle if successful.
  ///
  /// @param [in]  h        The plugin handle which is required to make this
  ///                       query
  /// @param [in] symbol    The symbol that needs to be searched within the
  ///                       shared object or DLL represented by the handle h
  ///
  /// @return <addr>        If the query is successful, the function returns
  ///                       a valid address
  /// @return nullptr       If the query fails, nullptr is returned
  ///
  xpti_plugin_handle_t loadLibrary(const char *path, std::string &error) {
    xpti_plugin_handle_t handle = 0;
#if defined(_WIN32) || defined(_WIN64)
    handle = LoadLibraryA(path);
    if (!handle) {
      error = getLastError();
    }
#else
    handle = dlopen(path, RTLD_LAZY);
    if (!handle) {
      error = getLastError();
    }
#endif
    return handle;
  }

  xpti::result_t unloadLibrary(xpti_plugin_handle_t h) {
    xpti::result_t fr = xpti::result_t::XPTI_RESULT_SUCCESS;
#if defined(_WIN32) || defined(_WIN64)
    if (!FreeLibrary(h)) {
      //  Error occurred while unloading the share object
      return xpti::result_t::XPTI_RESULT_FAIL;
    }
#else
    if (dlclose(h)) {
      //  Error occurred while unloading the share object
      return xpti::result_t::XPTI_RESULT_FAIL;
    }
#endif
    return fr;
  }

  /// @brief Checks is tracing has been enabled through XPTI_TRACE_ENABLE
  /// variable
  /// @details The environment variable XPTI_TRACE_ENABLE is checked to see if
  /// it is set. If it is not set, tracing is assumed to be enabled. If set,
  /// then "true" or "1" indicates enabled and "false" or "0" indicates
  /// disabled.
  /// @return bool    true if set to "true" or "1" and false if set to "false"
  /// or "0"
  bool checkTraceEnv() {
    std::string env = getEnvironmentVariable("XPTI_TRACE_ENABLE");
    if (env.empty()) {
      return true;
    }
    if (env == "true" || env == "1")
      return true;
    if (env == "false" || env == "0")
      return false;
    // If someone sets the variable to garbage, then we consider it as disabled
    return false;
  }
};

/// This is an implementation of a SpinLock synchronization primitive, that has
/// trivial constructor and destructor.
class SpinLock {
public:
  void lock() {
    while (MLock.test_and_set(std::memory_order_acquire))
      std::this_thread::yield();
  }
  void unlock() { MLock.clear(std::memory_order_release); }

private:
  std::atomic_flag MLock = ATOMIC_FLAG_INIT;
};

/// RAII-like helper to call a function upon exit from the scope.
///
/// This can be used to ensure that a specific XPTI API is called even if an
/// exception is thrown on code path. For convenience the function will only be
/// invoked if instrumentation is enabled.
struct finally {
  std::function<void()> MFunc;

  ~finally() {
    if (xptiTraceEnabled())
      MFunc();
  }
};

} // namespace utils

template <typename T>
inline result_t addMetadata(trace_event_data_t *Event, const std::string &Key,
                            const T &Data) {
  static_assert(std::is_trivially_copyable_v<T>,
                "T must be trivially copyable");
  static_assert(!std::is_same_v<T, const char *>);

  const uint8_t Type = [] {
    if (std::is_same_v<bool, T>) {
      return static_cast<uint8_t>(metadata_type_t::boolean);
    }
    if (std::numeric_limits<T>::is_integer &&
        std::numeric_limits<T>::is_signed) {
      return static_cast<uint8_t>(metadata_type_t::signed_integer);
    }
    if (std::numeric_limits<T>::is_integer &&
        !std::numeric_limits<T>::is_signed) {
      return static_cast<uint8_t>(metadata_type_t::unsigned_integer);
    }
    if (std::numeric_limits<T>::is_specialized &&
        !std::numeric_limits<T>::is_integer) {
      return static_cast<uint8_t>(metadata_type_t::floating);
    }

    return static_cast<uint8_t>(metadata_type_t::binary);
  }();

  object_id_t Value = xptiRegisterObject(reinterpret_cast<const char *>(&Data),
                                         sizeof(Data), Type);
  return xptiAddMetadata(Event, Key.c_str(), Value);
}

template <>
inline result_t addMetadata<std::string>(trace_event_data_t *Event,
                                         const std::string &Key,
                                         const std::string &Data) {
  const uint8_t Type = static_cast<uint8_t>(metadata_type_t::string);
  object_id_t Value = xptiRegisterObject(Data.c_str(), Data.size(), Type);
  return xptiAddMetadata(Event, Key.c_str(), Value);
}

template <>
inline result_t addMetadata<const char *>(trace_event_data_t *Event,
                                          const std::string &Key,
                                          const char *const &Data) {
  const uint8_t Type = static_cast<uint8_t>(metadata_type_t::string);
  object_id_t Value = xptiRegisterObject(Data, strlen(Data), Type);
  return xptiAddMetadata(Event, Key.c_str(), Value);
}

template <typename T>
inline std::pair<std::string_view, T>
getMetadata(const metadata_t::value_type &MD) {
  static_assert(std::is_trivially_copyable<T>::value,
                "T must be trivially copyable");

  object_data_t RawData = xptiLookupObject(MD.second);
  assert(RawData.size == sizeof(T));

  T Value = *reinterpret_cast<const T *>(RawData.data);

  const char *Key = xptiLookupString(MD.first);

  return std::make_pair(std::string_view(Key), Value);
}

template <>
inline std::pair<std::string_view, std::string>
getMetadata(const metadata_t::value_type &MD) {
  object_data_t RawData = xptiLookupObject(MD.second);

  std::string Value(RawData.data, RawData.size);

  const char *Key = xptiLookupString(MD.first);

  return std::make_pair(std::string_view(Key), Value);
}

template <>
inline std::pair<std::string_view, std::string_view>
getMetadata(const metadata_t::value_type &MD) {
  object_data_t RawData = xptiLookupObject(MD.second);

  std::string_view Value(RawData.data, RawData.size);

  const char *Key = xptiLookupString(MD.first);

  return std::make_pair(std::string_view(Key), Value);
}

inline std::string readMetadata(const metadata_t::value_type &MD) {
  object_data_t RawData = xptiLookupObject(MD.second);

  if (RawData.type == static_cast<uint8_t>(metadata_type_t::binary)) {
    return std::string("Binary data, size: ") + std::to_string(RawData.size);
  }

  if (RawData.type == static_cast<uint8_t>(metadata_type_t::boolean)) {
    bool Value = *reinterpret_cast<const bool *>(RawData.data);
    return Value ? "true" : "false";
  }

  if (RawData.type == static_cast<uint8_t>(metadata_type_t::signed_integer)) {
    if (RawData.size == 1) {
      auto I = *reinterpret_cast<const int8_t *>(RawData.data);
      return std::to_string(I);
    }
    if (RawData.size == 2) {
      auto I = *reinterpret_cast<const int16_t *>(RawData.data);
      return std::to_string(I);
    }
    if (RawData.size == 4) {
      auto I = *reinterpret_cast<const int32_t *>(RawData.data);
      return std::to_string(I);
    }
    if (RawData.size == 8) {
      auto I = *reinterpret_cast<const int64_t *>(RawData.data);
      return std::to_string(I);
    }
  }

  if (RawData.type == static_cast<uint8_t>(metadata_type_t::unsigned_integer)) {
    if (RawData.size == 1) {
      auto I = *reinterpret_cast<const uint8_t *>(RawData.data);
      return std::to_string(I);
    }
    if (RawData.size == 2) {
      auto I = *reinterpret_cast<const uint16_t *>(RawData.data);
      return std::to_string(I);
    }
    if (RawData.size == 4) {
      auto I = *reinterpret_cast<const uint32_t *>(RawData.data);
      return std::to_string(I);
    }
    if (RawData.size == 8) {
      auto I = *reinterpret_cast<const uint64_t *>(RawData.data);
      return std::to_string(I);
    }
  }

  if (RawData.type == static_cast<uint8_t>(metadata_type_t::floating)) {
    if (RawData.size == 4) {
      auto F = *reinterpret_cast<const float *>(RawData.data);
      return std::to_string(F);
    }
    if (RawData.size == 8) {
      auto F = *reinterpret_cast<const double *>(RawData.data);
      return std::to_string(F);
    }
  }

  if (RawData.type == static_cast<uint8_t>(metadata_type_t::string)) {
    return std::string(RawData.data, RawData.size);
  }

  return std::string("Unknown metadata type, size ") +
         std::to_string(RawData.size);
}

namespace framework {
constexpr uint16_t signal = (uint16_t)xpti::trace_point_type_t::signal;
constexpr uint16_t graph_create =
    (uint16_t)xpti::trace_point_type_t::graph_create;
constexpr uint16_t node_create =
    (uint16_t)xpti::trace_point_type_t::node_create;
constexpr uint16_t edge_create =
    (uint16_t)xpti::trace_point_type_t::edge_create;

class scoped_notify {
public:
  scoped_notify(const char *stream, uint16_t trace_type,
                xpti::trace_event_data_t *parent,
                xpti::trace_event_data_t *object, uint64_t instance,
                const void *user_data = nullptr)
      : m_object(object), m_parent(parent), m_stream_id(0),
        m_trace_type(trace_type), m_user_data(user_data), m_instance(instance) {
    if (xptiTraceEnabled()) {
      uint16_t open = m_trace_type & 0xfffe;
      m_stream_id = xptiRegisterStream(stream);
      xptiNotifySubscribers(m_stream_id, open, parent, object, instance,
                            m_user_data);
    }
  }

  ~scoped_notify() {
    if (xptiTraceEnabled()) {
      switch (m_trace_type) {
      case signal:
      case graph_create:
      case node_create:
      case edge_create:
        break;
      default: {
        uint16_t close = m_trace_type | 1;
        xptiNotifySubscribers(m_stream_id, close, m_parent, m_object,
                              m_instance, m_user_data);
      } break;
      }
    }
  }

private:
  xpti::trace_event_data_t *m_object, *m_parent;
  uint8_t m_stream_id;
  uint16_t m_trace_type;
  const void *m_user_data;
  uint64_t m_instance;
};

// --------------- Commented section of the code -------------
//
// github.com/bombela/backward-cpp/blob/master/backward.hpp
//
// Need to figure out the process for considering 3rd party
// code that helps with addressing the gaps when the developer
// doesn't opt-in.
//------------------------------------------------------------
// #include "backward.hpp"
// class backtrace_t {
// public:
//   backtrace_t(int levels = 2) {
//     m_st.load_here(levels);
//     m_tr.load_stacktrace(m_st);
//     m_parent = m_tr.resolve(m_st[1]);
//     m_curr = m_tr.resolve(m_st[0]);
//     if(m_parent.source.filename) {
//       m_payload = xpti::payload_t(m_curr.source.function,
//       m_parent.source.filename, m_parent.source.line, 0, m_curr.addr);
//     }
//     else {
//       m_packed_string = m_parent.source.function + std::string("::") +
//       m_curr.source.function; m_payload =
//       xpti::payload_t(m_curr.source.function, m_packed_string.c_str(),
//       m_curr.addr);
//     }
//   }
//
//   xpti::payload_t *payload() { return &m_payload;}
// private:
//   backward::StackTrace m_st;
//   backward::TraceResolver m_tr;
//   backward::ResolvedTrace m_curr, m_parent;
//   std::string m_packed_string;
//   xpti::payload_t m_payload;
// };

/// @brief Tracepoint data type allows the construction of Universal ID
/// @details The tracepoint data type builds on the payload data type by
/// combining the functionality of payload and xpti::makeEvent() to create the
/// unique Universal ID and stash it in the TLS for use by downstream layers in
/// the SW stack.
///
/// Usage:-
/// #ifdef XPTI_TRACE_ENABLED
///   xpti::payload_t p, *payload = &p;
/// #ifdef SYCL_TOOL_PROFILE
///   // sycl::detail::code_location cLoc =
///   // sycl::detail::code_location::current();
///   if(cLoc.valid())
///     p = xpti::payload_t(cLoc.functionname(), cLoc.fileName(),
///     cLoc.lineNumber(), cLoc.columnNumber(), codeptr);
///   else
///     p = xpti::payload_t(KernelInfo.funcName(), KernelInfo.sourceFileName(),
///     KernelInfo.lineNo(), KernelInfor.columnNo(), codeptr);
/// #else
///   xpti::framework::backtrace_t b;
///   payload = b.payload();
/// #endif
///   xpti::tracepoint_t t(payload);
/// #endif
///
///  See also: xptiTracePointTest in xpti_correctness_tests.cpp
class tracepoint_t {
public:
  // Constructor that makes calls to xpti API layer to register strings and
  // create the Universal ID that is stored in the TLS entry for lookup
  tracepoint_t(xpti::payload_t *p) : m_payload(nullptr), m_top(false) {
    if (p) {
      // We expect the payload input has been populated with the information
      // available at that time
      uint64_t uid = xptiGetUniversalId();
      if (uid != xpti::invalid_uid) {
        // We already have a parent SW layer that has a tracepoint defined
        m_payload = xptiQueryPayloadByUID(uid);
      } else {
        m_top = true;
        uid = xptiRegisterPayload(p);
        if (uid != xpti::invalid_uid) {
          xptiSetUniversalId(uid);
          m_payload = xptiQueryPayloadByUID(uid);
        }
      }
    }
  }
  ~tracepoint_t() {
    if (m_top) {
      xptiSetUniversalId(xpti::invalid_uid);
    }
  }

  // The payload object that is returned will have the UID object populated and
  // can be looked up in the xpti lookup APIs or be used to make an event.
  const payload_t *payload() { return m_payload; }

  uint64_t universal_id() {
    if (m_payload &&
        (m_payload->flags &
         static_cast<uint64_t>(xpti::payload_flag_t::HashAvailable))) {
      return m_payload->internal;
    } else {
      return xpti::invalid_uid;
    }
  }

private:
  /// The payload data structure that is prepared from code_location(),
  /// caller_callee string or kernel name/codepointer based on the opt-in flag.
  const payload_t *m_payload;
  bool m_top;
};
} // namespace framework
} // namespace xpti
