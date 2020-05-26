//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
#pragma once
#include <cstdint>
#include <sstream>

#include "xpti_data_types.h"
#include "xpti_trace_framework.h"

#if defined(_WIN32) || defined(_WIN64)
#include <string>
#include <strsafe.h>
#include <windows.h>
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
} // namespace utils

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
} // namespace framework
} // namespace xpti
