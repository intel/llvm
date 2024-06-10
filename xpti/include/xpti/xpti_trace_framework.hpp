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
#include <string>
#include <string_view>
#include <thread>

#include "xpti/xpti_data_types.h"
#include "xpti/xpti_trace_framework.h"

#if defined(_WIN32) || defined(_WIN64)
#include <string>
// Windows.h defines min and max macros, that interfere with C++ std::min and
// std::max. The following definition disables that feature.
#define NOMINMAX
#include <windows.h>
#undef NOMINMAX
// strsafe.h must be included after all other includes as per official
// documentation
#include <strsafe.h>
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

/// @class StringHelper
/// @brief A helper class for string manipulations.
///
/// This class provides methods for converting addresses to strings and for
/// creating strings that combine a prefix with an address.
///
class StringHelper {
public:
  /// @brief Converts an address to a hexadecimal string.
  ///
  /// @param address The address to be converted.
  /// @return The address as a hexadecimal string.
  ///
  template <class T> std::string addressAsString(T address) {
    std::stringstream ss;
    ss << std::hex << address;
    return ss.str();
  }

  /// @brief Creates a string that combines a prefix and an address.
  ///
  /// @param prefix The prefix to be used. If it is null, "unknown" is used.
  /// @param address The address to be included in the string.
  /// @return The combined string.
  ///
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

  /// @brief Creates a string that combines a prefix and an address.
  ///
  /// @param prefix The prefix to be used. If it is empty, "unknown" is used.
  /// @param address The address to be included in the string.
  /// @return The combined string.
  ///
  template <class T>
  std::string nameWithAddress(std::string &prefix, T address) {
    std::string coded_string;
    if (!prefix.empty())
      coded_string = prefix + "[" + addressAsString<T>(address) + "]";
    else
      coded_string = "unknown[" + addressAsString<T>(address) + "]";

    return coded_string;
  }

  /// @brief Creates a string that combines a prefix and an address.
  ///
  /// @param prefix The prefix to be used. If it is null, "unknown" is used.
  /// @param address The address to be included in the string.
  /// @return The combined string.
  ///
  std::string nameWithAddressString(const char *prefix, std::string &address) {
    std::string coded_string;

    if (prefix)
      coded_string = prefix;
    else
      coded_string = "unknown";

    coded_string += "[" + address + "]";
    return coded_string;
  }

  /// @brief Creates a string that combines a prefix and an address.
  ///
  /// @param prefix The prefix to be used. If it is empty, "unknown" is used.
  /// @param address The address to be included in the string.
  /// @return The combined string.
  ///
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
    UINT SavedMode = SetErrorMode(SEM_FAILCRITICALERRORS);
    // Exclude current directory from DLL search path
    if (!SetDllDirectoryA("")) {
      assert(false && "Failed to update DLL search path");
    }
    handle = LoadLibraryExA(path, NULL, NULL);
    if (!handle) {
      error = getLastError();
    }
    (void)SetErrorMode(SavedMode);
    if (!SetDllDirectoryA(nullptr)) {
      assert(false && "Failed to restore DLL search path");
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

/// @brief Adds metadata to a trace event.
///
/// This function template is used to add metadata of any trivially copyable
/// type to a trace event. The type of the metadata is determined at compile
/// time.
///
/// @tparam T The type of the metadata.
/// @param Event The trace event to which the metadata is added.
/// @param Key The key of the metadata.
/// @param Data The metadata to be added.
/// @return The result of the metadata addition operation.
///
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

/// @brief Retrieves metadata from a metadata object.
///
/// This function template is used to retrieve metadata of any trivially
/// copyable type from a metadata object. The type of the metadata is determined
/// at compile time.
///
/// @tparam T The type of the metadata.
/// @param MD The metadata object from which the metadata is retrieved.
/// @return A pair consisting of the metadata key and the metadata.
///
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

/// @brief Retrieves string metadata from a metadata object.
///
/// This function template specialization is used to retrieve string metadata
/// from a metadata object.
///
/// @param MD The metadata object from which the metadata is retrieved.
/// @return A pair consisting of the metadata key and the metadata.
///
template <>
inline std::pair<std::string_view, std::string>
getMetadata(const metadata_t::value_type &MD) {
  object_data_t RawData = xptiLookupObject(MD.second);

  std::string Value(RawData.data, RawData.size);

  const char *Key = xptiLookupString(MD.first);

  return std::make_pair(std::string_view(Key), Value);
}

/// @brief Retrieves string view metadata from a metadata object.
///
/// This function template specialization is used to retrieve string view
/// metadata from a metadata object.
///
/// @param MD The metadata object from which the metadata is retrieved.
/// @return A pair consisting of the metadata key and the metadata.
///
template <>
inline std::pair<std::string_view, std::string_view>
getMetadata(const metadata_t::value_type &MD) {
  object_data_t RawData = xptiLookupObject(MD.second);

  std::string_view Value(RawData.data, RawData.size);

  const char *Key = xptiLookupString(MD.first);

  return std::make_pair(std::string_view(Key), Value);
}

/// @brief Reads metadata from a metadata object.
///
/// This function is used to read metadata from a metadata object and convert it
/// to a string. The type of the metadata is determined at runtime.
///
/// @param MD The metadata object from which the metadata is read.
/// @return A string representation of the metadata.
///
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
// Scoped class that assists in stashing a tuple and clearing it when it is pout
// of scope
class stash_tuple {
public:
  stash_tuple(const char *key, uint64_t value) : m_stashed(false) {
    m_stashed =
        (xptiStashTuple(key, value) == xpti::result_t::XPTI_RESULT_SUCCESS);
  }
  ~stash_tuple() {
    if (m_stashed) {
      xptiUnstashTuple();
    }
  }

private:
  bool m_stashed;
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

/// @class tracepoint_scope_t
/// @brief This class is used to manage the lifecycle of a tracepoint.
///
/// It provides methods to initialize, notify, and add metadata to a tracepoint.
/// The class also manages the scope of a tracepoint, ensuring that the
/// tracepoint is properly registered and released.
///
/// Example usage:-
/// {
///   xpti::framework::tracepoint_scope_t TS(); // Will send a self notification
///   TS.scopeNotify((uint16_t)xpti::trace_point_type_t::task_begin,
///   "my_task_foo");
///   TS.addMetadata([&](auto TEvent) {
///     xpti::addMetadata(TEvent, "memory_size", Count);
///     xpti::addMetadata(TEvent, "queue_id", MQueueID);
///   });
/// } // Will send a second notification with the task_end type
///
class tracepoint_scope_t {
public:
  ///
  /// @brief Constructor that initializes a tracepoint with a file name,
  /// function name, line, and column.
  ///
  /// @param fileName The name of the file where the tracepoint is located.
  /// @param funcName The name of the function where the tracepoint is located.
  /// @param line The line number where the tracepoint is located.
  /// @param column The column number where the tracepoint is located.
  /// @param selfNotify A boolean indicating whether the tracepoint should
  /// notify its scope.
  /// @param callerFuncName The name of the function that is calling this
  /// constructor.
  ///
  tracepoint_scope_t(const char *fileName, const char *funcName, int line,
                     int column, bool selfNotify = false,
                     const char *callerFuncName = __builtin_FUNCTION())
      : MTop(false), MSelfNotify(selfNotify), MCallerFuncName(callerFuncName) {
    if (!xptiTraceEnabled())
      return;

    MDefaultStreamId = MStreamId = xptiGetDefaultStreamID();
    MData = xptiGetTracepointScopeData();
    if (!MData.isValid()) {
      // No scope data has been set, so we will initiate the scope here by
      // registering the payload
      payload_t tpPayload(funcName, fileName, line, column);
      // If the incomming information to create the payload is invalid, we
      // will create a payload with the caller function name
      if (!tpPayload.isValid()) {
        tpPayload = payload_t(callerFuncName, nullptr, 0, 0);
      }
      init(tpPayload);
    }
    selfNotifyBegin();
  }

  /// @brief Constructor that initializes a tracepoint with a payload_t object
  ///
  /// @param tpPayload The payload that results in a UID and trace event.
  /// @param selfNotify A boolean indicating whether the tracepoint should
  /// notify its scope.
  /// @param callerFuncName The name of the function that is calling this
  /// constructor.
  ///
  tracepoint_scope_t(payload_t &tpPayload, bool selfNotify = false,
                     const char *callerFuncName = __builtin_FUNCTION())
      : MTop(false), MSelfNotify(selfNotify), MCallerFuncName(callerFuncName) {
    if (!xptiTraceEnabled())
      return;

    MDefaultStreamId = MStreamId = xptiGetDefaultStreamID();
    MData = xptiGetTracepointScopeData();
    if (!MData.isValid()) {
      // No scope data has been set, so we will initiate the scope here by using
      // the payload object. If the object has already been registered, the
      // return value will have the same UID for the payload, but with an
      // updated instance ID
      // If the incomming payload  is invalid, create a new one with the caller
      // function name
      if (!tpPayload.isValid()) {
        payload_t payload = payload_t(callerFuncName, nullptr, 0, 0);
        init(payload);
      } else {
        init(tpPayload);
      }
    }
    selfNotifyBegin();
  }

  ///
  /// @brief Constructor that initializes a tracepoint
  ///
  /// @details This constructor initializes a tracepoint with no payload. The
  /// object in this case attempts to get the required information from TLS, if
  /// it exists. If it doesn't, the tracepoint is not registered and no self
  /// notification is sent out, even if it is requested.
  ///
  /// @param selfNotify A boolean indicating whether the tracepoint should
  /// notify its scope.
  /// @param callerFuncName The name of the function that is calling this
  /// constructor.
  ///
  tracepoint_scope_t(bool selfNotify = false,
                     const char *callerFuncName = __builtin_FUNCTION())
      : MTop(false), MSelfNotify(selfNotify), MCallerFuncName(callerFuncName) {
    if (!xptiTraceEnabled())
      return;

    MDefaultStreamId = MStreamId = xptiGetDefaultStreamID();
    MData = xptiGetTracepointScopeData();
    selfNotifyBegin();
  }

  /// @brief  Disable copy construction
  tracepoint_scope_t(const tracepoint_scope_t &rhs) = delete;
  /// @brief  Disable assignment
  tracepoint_scope_t &operator=(const tracepoint_scope_t &rhs) = delete;

  /// @brief Initializes the tracepoint with the provided payload.
  ///
  /// This function registers the payload and prepares the tracepoint data. It
  /// also creates the tracepoint scope data and the trace event for this
  /// instance of the payload. If this is the top-level tracepoint, it will set
  /// the top-level flag to true so the trace even created can be released when
  /// the tracepoint goes out of scope.
  ///
  /// @param tpPayload The payload to be registered and used for preparing the
  /// tracepoint data.
  ///
  void init(payload_t &tpPayload) {
    // Register the payload and prepare the tracepoint data. The function
    // returns a UID, associated payload and trace event
    MData = xptiRegisterPayloadAndPrepareTracepointData(tpPayload);
    // Set the tracepoint scope with the prepared data so all nested functions
    // will have access to it; this call also sets the Universal ID separately
    // for backward compatibility.
    xptiSetTracepointScopeData(MData);
    // Set the trace event for this tracepoint; all notifications will use this
    MTraceEvent = MData.event;
    // Set the top flag to true, indicating that this is the top-level
    // tracepoint.
    MTop = true;
  }

  /// @brief Destructor for the tracepoint_scope_t class.
  ///
  /// @details This function is responsible for cleaning up when an instance of
  /// tracepoint_scope_t is no longer needed. It notifies subscribers if scoped
  /// notification is enabled and the trace type is set. If self notification
  /// flag is set, it notifies the the end of the tracepoint scope. If this
  /// instance is the top-level tracepoint, it releases the event, unsets the
  /// tracepoint scope data, resets the data member, and sets the top flag to
  /// false.
  ///
  ~tracepoint_scope_t() {
    // If scoped notification is enabled and the trace type is set, notify
    // subscribers with the closing trace type by setting the LSB to 1 so it
    // will correspond to the the one specified in scopedNotify(). We will check
    // if the trace type has a scope defined before notifying.
    if (MScopedNotify && MTraceType && isTraceTypeScoped()) {

      MTraceType = MTraceType | 1;
      if (xptiCheckTraceEnabled(MStreamId, MTraceType))
        xptiNotifySubscribers(MStreamId, MTraceType, MParentEvent, MTraceEvent,
                              MScopedCorrelationId, MScopedUserData);
    }
    // If self notify flag is set, notify the end of the scope
    selfNotifyEnd();
    // If this is the top-level tracepoint, perform additional cleanup.
    if (MTop) {
      // Release the event created since this instance of the payload is going
      // out of scope
      xptiReleaseEvent(MData.event);
      // Reset TLS data to invalid
      xptiUnsetTracepointScopeData();
      // Clear internal state
      MData = xpti::trace_point_data_t();
      MTop = false;
    }
  }

  /// @brief Returns the unique identifier (UID) of the tracepoint.
  ///
  /// This function returns a reference to the UID of the tracepoint, which is
  /// stored in the tracepoint's data. This is created when the tracepoint is
  /// created using the payload or inherited through the TLS
  ///
  /// @return A reference to the UID of the tracepoint.
  ///
  xpti::uid_t &uid() { return MData.uid; }

  /// @brief Returns the payload of the tracepoint.
  ///
  /// This function returns a pointer to the payload of the tracepoint, which is
  /// which corresponds to the UID in the trace point data.
  ///
  /// @return A pointer to the payload of the tracepoint.
  ///
  xpti::payload_t *payload() { return MData.payload; }

  /// @brief Returns the trace event data of the tracepoint.
  ///
  /// This function returns a pointer to the trace event data of the tracepoint
  /// instance.
  ///
  /// @return A pointer to the trace event data of the tracepoint.
  ///
  xpti::trace_event_data_t *traceEvent() { return MTraceEvent; }

  /// @brief Returns the stream ID for the tracepoint.
  ///
  /// This function returns stream ID for the strem which will be used to
  /// publish events for this tracepoint instance.
  ///
  /// @return The stream ID.
  ///
  uint8_t streamId() { return MStreamId; }

  /// @brief Sets the stream for the tracepoint scoped notification
  ///
  /// This method registers a stream with the given name and sets the stream ID
  /// for the tracepoint. If tracing is not enabled, the method does nothing.
  ///
  /// @param streamName The name of the stream to be registered.
  /// @return A reference to this tracepoint_scope_t object.
  ///
  tracepoint_scope_t &stream(const char *streamName) {
    // If tracing is not enabled, don't do anything
    if (xptiTraceEnabled()) {
      MStreamId = xptiRegisterStream(streamName);
    }
    return *this;
  }

  /// @brief Sets the trace type for the tracepoint scoped notification
  ///
  /// This method sets the trace type for the tracepoint. The trace type is an
  /// enum value that indicates the type of the tracepoint (e.g.,
  /// function_begin, function_end, etc.).
  ///
  /// @param type The trace type to be set for the tracepoint.
  /// @return A reference to this tracepoint_scope_t object.
  ///
  tracepoint_scope_t &traceType(xpti::trace_point_type_t type) {
    MTraceType = (uint16_t)type;
    return *this;
  }

  /// @brief Sets the event type for the tracepoint.
  ///
  /// This method sets the event type for the tracepoint if tracing is enabled.
  /// The event type is an enum value that indicates the type of the event
  /// (e.g., algorithm, barrier, etc.). If a trace event exists, its event type
  /// is also updated.
  ///
  /// @param type The event type to be set for the tracepoint.
  /// @return A reference to this tracepoint_scope_t object.
  ///
  tracepoint_scope_t &eventType(xpti::trace_event_type_t type) {
    if (xptiTraceEnabled()) {
      MEventType = (uint16_t)type;
      if (MTraceEvent)
        MTraceEvent->event_type = MEventType;
    }
    return *this;
  }

  /// @brief Sets the activity type for the tracepoint.
  ///
  /// This method sets the activity type for the tracepoint if tracing is
  /// enabled. The activity type is an enum value that indicates the type of the
  /// activity (e.g., active, overhead, etc.). If a trace event exists, its
  /// activity type is also updated.
  ///
  /// @param type The activity type to be set for the tracepoint.
  /// @return A reference to this tracepoint_scope_t object.
  ///
  tracepoint_scope_t &activityType(xpti::trace_activity_type_t type) {
    if (xptiTraceEnabled()) {
      MActivityType = type;
      if (MTraceEvent)
        MTraceEvent->activity_type = (uint16_t)MActivityType;
    }
    return *this;
  }

  /// @brief Sets the parent event for the tracepoint.
  ///
  /// This method sets the parent event for the tracepoint if tracing is
  /// enabled. The parent event is another trace event that is logically the
  /// parent of this tracepoint's event.
  ///
  /// @param event A pointer to the parent trace event data.
  /// @return A reference to this tracepoint_scope_t object.
  ///
  tracepoint_scope_t &parentEvent(xpti::trace_event_data_t *event) {
    if (xptiTraceEnabled()) {
      MParentEvent = event;
    }
    return *this;
  }

  /// @brief Notifies subscribers about the tracepoint.
  ///
  /// This method sends a notification to all subscribers about the tracepoint
  /// if tracing is enabled. This method typically notifies about an instance in
  /// the trace point scope or information about an object
  ///
  /// @param userData A pointer to the user data to be included in the
  /// notification.
  ///
  void notify(const void *userData) {
    // If tracing is not enabled, don't notify
    if (!xptiCheckTraceEnabled(MStreamId, MTraceType))
      return;

    xptiNotifySubscribers(MStreamId, MTraceType, MParentEvent, MTraceEvent,
                          xptiGetUniqueId(), userData);
  }

  /// @brief Notifies subscribers about the a user-defined scope.
  ///
  /// @details This method sends a scoped notification to all subscribers about
  /// the tracepoint if tracing is enabled and the tracepoint data is valid. The
  /// trace type is masked to ensure it is even (i.e., a begin type). If the
  /// trace type is enabled for the stream, the user data and correlation ID are
  /// communictaed to the subscribers.
  ///
  /// @param traceType The trace type to be used for the notification.
  /// @param userData A pointer to the user data to be included in the
  /// notification.
  ///
  void scopedNotify(uint16_t traceType, const void *userData) {
    // If tracing is not enabled, don't notify
    if (!xptiTraceEnabled() || !MData.isValid())
      return;

    MTraceType = traceType & 0xfffe;
    MScopedNotify = true;
    if (xptiCheckTraceEnabled(MStreamId, traceType) && MTraceType) {
      MScopedUserData = const_cast<void *>(userData);
      MScopedCorrelationId = xptiGetUniqueId();
      // Notify all subscribers with the tracepoint related details
      xptiNotifySubscribers(MStreamId, traceType, MParentEvent, MTraceEvent,
                            MScopedCorrelationId, userData);
    }
  }

  /// @brief Adds metadata to the trace event.
  ///
  /// This method adds metadata to the trace event if tracing is enabled and the
  /// trace event is valid. The metadata is added by calling the provided
  /// callback function with the trace event data.
  ///
  /// @param Callback A function that takes a pointer to the trace event data
  /// and adds the desired metadata.
  /// @return A reference to this tracepoint_scope_t object.
  ///
  tracepoint_scope_t &
  addMetadata(const std::function<void(xpti::trace_event_data_t *)> &Callback) {
    if (xptiCheckTraceEnabled(MStreamId, MTraceType) && MTraceEvent) {
      Callback(MTraceEvent);
    }
    return *this;
  }

private:
  void selfNotifyEnd() {
    // selfNotify() is supposed to capture the scope of all the tracepoints we
    // have enbled; if we need to limit the notifications only to when the trace
    // scope has valid TLS data, we can add the follwoing IF block if
    // (!m_data.isValid())
    //   return;
    uint16_t traceType = (uint16_t)xpti::trace_point_type_t::function_end;
    if (MSelfNotify && MDefaultStreamId &&
        xptiCheckTraceEnabled(MDefaultStreamId, traceType)) {
      const void *UserData = static_cast<const void *>(MCallerFuncName);
      xptiNotifySubscribers(MDefaultStreamId, traceType, nullptr, MTraceEvent,
                            MCorrelationId, UserData);
    }
  }

  void selfNotifyBegin() {
    // selfNotify() is supposed to capture the scope of all the tracepoints we
    // have enbled; if we need to limit the notifications only to when the trace
    // scope has valid TLS data, we can add the follwoing IF block if
    // (!m_data.isValid())
    //   return;
    uint16_t traceType = (uint16_t)xpti::trace_point_type_t::function_begin;
    if (MSelfNotify && MDefaultStreamId &&
        xptiCheckTraceEnabled(MDefaultStreamId, traceType)) {
      MCorrelationId = xptiGetUniqueId();
      const void *UserData = static_cast<const void *>(MCallerFuncName);
      xptiNotifySubscribers(MDefaultStreamId, traceType, nullptr, MTraceEvent,
                            MCorrelationId, UserData);
    }
  }

  bool isTraceTypeScoped() {
    if (MTraceType != (uint16_t)xpti::trace_point_type_t::signal &&
        MTraceType != (uint16_t)xpti::trace_point_type_t::graph_create &&
        MTraceType != (uint16_t)xpti::trace_point_type_t::node_create &&
        MTraceType != (uint16_t)xpti::trace_point_type_t::edge_create &&
        MTraceType != (uint16_t)xpti::trace_point_type_t::queue_create &&
        MTraceType != (uint16_t)xpti::trace_point_type_t::queue_destroy &&
        MTraceType != (uint16_t)xpti::trace_point_type_t::diagnostics)
      return true;
    else
      return false;
  }

  /// Indicates if this is the top-level tracepoint
  bool MTop = false;
  /// Indicates if the tracepoint should notify the scope of the tracepoint
  bool MSelfNotify = true;
  /// Indicates if scoped notification is enabled
  bool MScopedNotify = false;
  /// Stores the tracepoint data from creating the tracepoint or from
  /// inheritance
  trace_point_data_t MData;
  /// Stores the name of the function that contains the tracepoint and self
  /// notification will use this information during notification.
  const char *MCallerFuncName;
  /// Stores the correlation ID for the tracepoint's scope notification.
  uint64_t MCorrelationId;
  /// Stores the correlation ID for the scoped tracepoint which is different
  /// from self notification
  uint64_t MScopedCorrelationId;
  /// Stores the ID of the stream
  uint8_t MStreamId = 0;
  /// Stores the ID of the default stream use for self notification; the
  /// system sets the default stream ID at the start of the program
  uint8_t MDefaultStreamId = 0;
  /// Stores the user data for the scoped tracepoint notification
  void *MScopedUserData = nullptr;
  /// Stores the type of the tracepoint for scoped notification
  uint16_t MTraceType = (uint16_t)xpti::trace_point_type_t::function_begin;
  /// Stores the type of the event to describe the scoped notifications
  uint16_t MEventType = (uint16_t)xpti::trace_event_type_t::algorithm;
  /// Stores the type of the activity the scoped notification region performing
  /// - active, sleep, barrier, lock, overhead etc
  xpti::trace_activity_type_t MActivityType =
      xpti::trace_activity_type_t::active;
  /// Stores a pointer to the trace event data
  trace_event_data_t *MTraceEvent = nullptr;
  /// Stores a pointer to the parent event data, if provided
  trace_event_data_t *MParentEvent = nullptr;
};

/// @class notify_scope_t
/// @brief A class that provides scope-based notification for tracing.
///
/// This class is used to send notifications to subscribers when entering and
/// exiting a scope. The notifications are only sent if tracing is enabled for
/// the given stream and trace type. This object also uses appropriate defaults
/// for the trace type and stream ID based on the most frequently used types and
/// streams respectively.
///
class notify_scope_t {
public:
  /// @brief Constructs a notify_scope_t object.
  ///
  /// The constructor checks if tracing is enabled for the given stream and
  /// trace type. If tracing is enabled, it generates a unique correlation ID
  /// for use by the notification calls.
  ///
  /// @param streamId The ID of the stream.
  /// @param traceEvent The trace event data.
  /// @param UserData The user data.
  /// @param traceType The type of the trace event. Defaults to function_begin.
  ///
  notify_scope_t(uint8_t streamId, xpti::trace_event_data_t *traceEvent,
                 const char *UserData,
                 uint16_t traceType = (uint16_t)
                     xpti::trace_point_type_t::function_begin)
      : MTraceEvent(traceEvent), MUserData(UserData), MStreamId(streamId),
        MTraceType(traceType), MScopedNotify(false) {
    if (!xptiCheckTraceEnabled(MStreamId, MTraceType))
      return;

    MCorrelationId = xptiGetUniqueId();
  }

  /// @brief Sends a notification to subscribers.
  ///
  /// This method checks if tracing is enabled for the stream and trace type.
  /// If tracing is enabled, it sends a notification to subscribers. This call
  /// is used for trace types that do not have a begin and end scopes.
  ///
  /// @return A reference to this object.
  ///
  notify_scope_t &notify() {
    if (!xptiCheckTraceEnabled(MStreamId, MTraceType))
      return *this;

    uint16_t TraceType = MTraceType & 0xfffe;
    xptiNotifySubscribers(MStreamId, TraceType, nullptr, MTraceEvent,
                          MCorrelationId, static_cast<const void *>(MUserData));
    return *this;
  }

  /// @brief Sends a scoped notification to subscribers.
  ///
  /// This method checks if tracing is enabled for the stream and trace type.
  /// If tracing is enabled, it sends a scoped notification to subscribers and
  /// sets the scoped notify flag so the end of scope notification can be sent
  /// out when the object is destroyed.
  ///
  /// @return A reference to this object.
  ///
  notify_scope_t &scopedNotify() {
    if (!xptiCheckTraceEnabled(MStreamId, MTraceType))
      return *this;

    uint16_t TraceType = MTraceType & 0xfffe;
    MScopedNotify = true;
    xptiNotifySubscribers(MStreamId, TraceType, nullptr, MTraceEvent,
                          MCorrelationId, static_cast<const void *>(MUserData));
    return *this;
  }

  /// @brief The destructor of the notify_scope_t class.
  ///
  /// The destructor checks if tracing is enabled for the stream and trace type.
  /// If tracing is enabled, it sends an end-of-scope notification to
  /// subscribers.
  ///
  ~notify_scope_t() {
    if (!MScopedNotify || !xptiCheckTraceEnabled(MStreamId, MTraceType))
      return;

    uint16_t TraceType = MTraceType | 1;
    xptiNotifySubscribers(MStreamId, TraceType, nullptr, MTraceEvent,
                          MCorrelationId, static_cast<const void *>(MUserData));
  }

private:
  /// The trace event for which the notifications are being sent out
  xpti::trace_event_data_t *MTraceEvent;
  /// The user data to be sent out with the notification
  const char *MUserData;
  /// The stream with which the notification is associated
  uint8_t MStreamId;
  /// The type of the trace event: function_begin, signal, etc
  uint16_t MTraceType;
  /// The correlation ID for the notification begin/end scope calls
  uint64_t MCorrelationId;
  /// If scoped notification is desired
  bool MScopedNotify;
};

} // namespace framework
} // namespace xpti

/// @def XPTI_LW_TRACE(streamId, traceEvent)
/// @brief A macro that creates a lightweight trace scope and sends a scoped
/// notification.
///
/// This macro creates an instance of xpti::framework::notify_scope_t and calls
/// its scopedNotify method. The scopedNotify method sends a notification to
/// subscribers when entering and exiting the scope.
///
/// @param streamId The ID of the stream.
/// @param traceEvent The trace event for which notiofications are desired.
///
#define XPTI_LW_TRACE(streamId, traceEvent)                                    \
  xpti::framework::notify_scope_t LWTrace(streamId, traceEvent,                \
                                          __builtin_FUNCTION())                \
      .scopedNotify();

/// @def XPTI_SET_TRACE_SCOPE(fileN, funcN, lineN, colN, traceType, traceEv)
/// @brief A macro that sets a trace scope and sends a scoped notification.
///
/// This macro creates an instance of xpti::framework::tracepoint_scope_t and
/// calls its scopedNotify method. The scopedNotify method sends a notification
/// to subscribers when entering and exiting the scope. In this case, a new
/// scope is created using the code location information. The payload containing
/// the code location information is registered and an associated trace event
/// created which will be used for the scoped notification calls
///
/// @param fileN The name of the file.
/// @param funcN The name of the function.
/// @param lineN The line number.
/// @param colN The column number.
/// @param traceType The type of the trace event.
/// @param traceEv The trace event data.
///
#define XPTI_SET_TRACE_SCOPE(fileN, funcN, lineN, colN, traceType)             \
  xpti::framework::tracepoint_scope_t TP(fileN, funcN, lineN, colN)            \
      .scopedNotify(traceType, static_cast<const void *>(funcN))

/// @def XPTI_USE_TRACE_SCOPE(self)
/// @brief A macro that creates a trace scope for the current function and sends
/// a scoped notification.
///
/// This macro creates an instance of xpti::framework::tracepoint_scope_t and
/// sends a scoped notification to subscribers with the user data for the
/// notifications set to the name of the current function.
///
/// @param self A pointer to the current function.
///
#define XPTI_USE_TRACE_SCOPE() xpti::framework::tracepoint_scope_t TP(true);
