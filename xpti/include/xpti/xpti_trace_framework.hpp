//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
#pragma once

#include "xpti/xpti_data_types.h"
#include "xpti/xpti_trace_framework.h"
#include <atomic>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>

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

/// @brief Creates a unique 128-bit identifier (UID) for tracking entities.
///
/// This function combines file and function identifiers with line and column
/// information to generate a unique identifier. The UID is composed of two
/// 64-bit parts: the first part (p1) combines the file and function IDs, and
/// the second part (p2) combines the column and line numbers. An initial
/// instance count of 1 is set, indicating the creation of a new UID.
///
/// @param FileID The unique identifier for the file.
/// @param FuncID The unique identifier for the function.
/// @param Line The line number where the entity is located.
/// @param Col The column number where the entity is located.
/// @return A `xpti::uid128_t` structure representing the unique identifier.
///
inline xpti::uid128_t make_uid128(uint64_t FileID, uint64_t FuncID, int Line,
                                  int Col) {
  xpti::uid128_t UID;
  UID.p1 = (FileID << 32) | FuncID;
  UID.p2 = ((uint64_t)Col << 32) | Line;
  UID.instance = 0;
  return UID;
}

inline bool is_valid_uid(const xpti::uid128_t &UID) {
  return (UID.p1 != 0 || UID.p2 != 0) && UID.instance > 0;
}

inline bool is_valid_payload(const xpti::payload_t *Payload) {
  if (!Payload)
    return false;
  else
    return (Payload->flags != 0) &&
           ((Payload->flags &
                 static_cast<uint64_t>(payload_flag_t::SourceFileAvailable) ||
             (Payload->flags &
              static_cast<uint64_t>(payload_flag_t::NameAvailable))));
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
  scoped_notify(uint8_t stream_id, uint16_t trace_type,
                xpti::trace_event_data_t *parent,
                xpti::trace_event_data_t *object, uint64_t instance,
                const void *user_data = nullptr)
      : m_object(object), m_parent(parent), m_stream_id(stream_id),
        m_trace_type(trace_type), m_user_data(user_data), m_instance(instance) {
    if (!xptiTraceEnabled())
      return;
    uint16_t open = m_trace_type & 0xfffe;
    xptiNotifySubscribers(m_stream_id, open, parent, object, instance,
                          m_user_data);
  }

  ~scoped_notify() {
    if (xptiTraceEnabled())
      return;
    switch (m_trace_type) {
    case signal:
    case graph_create:
    case node_create:
    case edge_create:
      break;
    default: {
      uint16_t close = m_trace_type | 1;
      xptiNotifySubscribers(m_stream_id, close, m_parent, m_object, m_instance,
                            m_user_data);
    } break;
    }
  }

private:
  xpti::trace_event_data_t *m_object, *m_parent;
  uint8_t m_stream_id;
  uint16_t m_trace_type;
  const void *m_user_data;
  uint64_t m_instance;
};

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

/// @class uid_object_t
/// @brief Represents an object with a unique identifier.
///
/// This class encapsulates a unique identifier (UID) and provides various
/// constructors, assignment operators, comparison operators, and utility
/// methods to work with the UID. The UID is represented by an `xpti::uid128_t`
/// structure, which includes fields for different parts of the UID such as file
/// ID, function ID, line number, and column number, along with an instance
/// identifier.
///
/// The class offers functionality to:
/// - Initialize a `uid_object_t` object with a specific UID or default values.
/// - Copy and assign `uid_object_t` objects.
/// - Compare `uid_object_t` objects based on their UIDs.
/// - Check the validity of the UID.
/// - Extract specific parts of the UID, such as file ID, function ID, line
/// number, and column number.
/// - Retrieve the complete UID or instance identifier.
///
/// @note The comparison operators (`==`, `!=`, `<`) consider different parts of
/// the UID for comparison, but the instance identifier is not considered in
/// equality or inequality comparisons.
///
class uid_object_t {
public:
  /// @brief Constructs a uid_object_t object with a given UID.
  ///
  /// This constructor initializes the object with an existing UID provided as a
  /// parameter. It's useful when you have a UID ready and want to create a
  /// uid_object_t object that represents this UID and utilize all the helper
  /// features.
  ///
  /// @param UId A reference to an existing xpti::uid128_t object that will be
  /// used to initialize this object.
  ///
  uid_object_t(xpti::uid128_t &UId) : MUId(UId) {}

  /// @brief Default constructor that initializes a uid_object_t object with
  /// default values.
  ///
  /// This constructor sets the internal UID (MUId) to a default state where all
  /// parts of the UID (p1, p2, and instance) are set to 0. It's useful for
  /// creating a uid_object_t object when no UID is available at the time of
  /// creation, representing an uninitialized or null state. This null state UId
  /// is always considered invalid.
  ///
  uid_object_t() {
    MUId.p1 = 0;
    MUId.p2 = 0;
    MUId.instance = 0;
  };

  /// @brief Copy constructor for creating a uid_object_t object as a copy of
  /// another.
  ///
  /// This constructor creates a new uid_object_t object as an exact copy of an
  /// existing one. It copies the UID (MUId) from the provided object (Rhs) to
  /// the new object. This is useful for duplicating uid_object_t objects,
  /// ensuring the new object has the same UID as the one it's copied from.
  ///
  /// @param Rhs A reference to the uid_object_t object to be copied.
  ///
  uid_object_t(const uid_object_t &Rhs) { MUId = Rhs.MUId; }

  /// @brief Assigns the value of one uid_object_t to another.
  ///
  /// This copy assignment operator allows for the assignment of one
  /// uid_object_t object to another, making the left-hand side object a copy of
  /// the right-hand side object. It achieves this by copying the UID (MUId)
  /// from the right-hand side object (Rhs) to the left-hand side object. This
  /// operator is essential for object management in C++, enabling the copying
  /// of objects, either explicitly or when objects are passed by value or
  /// returned from functions.
  ///
  /// @param Rhs A reference to the uid_object_t object to be copied. This is
  /// the source object.
  /// @return Returns a reference to the current object after copying, allowing
  /// for chain assignments.
  ///
  uid_object_t &operator=(const uid_object_t &Rhs) {
    MUId = Rhs.MUId;
    return *this;
  }

  /// @brief Assigns a new UID to the uid_object_t object.
  ///
  /// This assignment operator allows for the direct assignment of a
  /// `xpti::uid128_t` UID to a `uid_object_t` object. It replaces the current
  /// UID
  /// (`MUId`) of the `uid_object_t` object with the UID provided in `Rhs`. This
  /// operation is useful when you need to update the UID of an existing
  /// `uid_object_t` object without creating a new instance.
  ///
  /// @param Rhs A constant reference to a `xpti::uid128_t` object representing
  /// the new UID to be assigned.
  /// @return Returns a reference to the current `uid_object_t` object to allow
  /// for chain assignments.
  ///
  uid_object_t &operator=(const xpti::uid128_t &Rhs) {
    MUId = Rhs;
    return *this;
  }

  /// @brief Overload of operator== to compare two xpti::uid128_t objects.
  ///
  /// This operator overload allows for the comparison of two xpti::uid128_t
  /// objects. Two uid_t objects are considered equal if their p1 and p2 fields
  /// are equal. The p1 field contains the combined file ID and function ID, and
  /// the p2 field contains the combined line number and column number. The
  /// instance field is not taken into account for equality.
  ///
  /// @param rhs The right-hand side xpti::uid128_t object in the comparison.
  /// @return Returns true if the p1 and p2 fields of the two xpti::uid128_t
  /// objects are equal, and false otherwise.
  bool operator==(const uid_object_t &Rhs) const {
    return (MUId.p1 == Rhs.MUId.p1 && MUId.p2 == Rhs.MUId.p2);
  }

  /// @brief Overload of operator!= to compare two xpti::uid128_t objects.
  ///
  /// This operator overload allows for the comparison of two xpti::uid128_t
  /// objects. Two uid_t objects are considered not equal if their p1 and p2
  /// fields are not equal. The instance field is not taken into account for
  /// inequality.
  ///
  /// @param rhs The right-hand side xpti::uid128_t object in the comparison.
  /// @return Returns true if any of the p1 and p2 fields of the two
  /// xpti::uid128_t objects are not equal, and false if noth parts are equal.
  bool operator!=(const uid_object_t &Rhs) const {
    return (MUId.p1 != Rhs.MUId.p1 && MUId.p2 != Rhs.MUId.p2);
  }
  /// @brief Compares two `uid_t` objects.
  ///
  /// This operator overload allows for the comparison of two `uid_t` objects.
  /// A `uid_t` object is considered less than another if its `p1` field is
  /// less than the other's, or if their `p1` fields are equal and its `p2`
  /// field is less than the other's.
  ///
  /// @param rhs The right-hand side `uid_t` object in the comparison.
  /// @return Returns true if the left-hand side `uid_t` object is less than
  /// the right-hand side one, and false otherwise.
  bool operator<(const uid_object_t &Rhs) const {
    if (MUId.p1 < Rhs.MUId.p1)
      return true;
    if (MUId.p1 == Rhs.MUId.p1 && MUId.p2 < Rhs.MUId.p2)
      return true;
    return false;
  }

  /// @brief Checks if the uid_t member variable is valid.
  ///
  /// This method checks if the uid_t object is valid by checking if any of the
  /// hash parts (`p1`, `p2`) or the `instance` member is not zero. The
  /// `instance` member is a unique identifier for the instance of the uid_t
  /// object. If any of these members is not zero, it means that the uid_t
  /// object is considered valid.
  ///
  /// @return Returns true if the uid_t object is valid (i.e., if any of the
  /// hash parts and the `instance` member is not zero), and false otherwise.
  ///
  bool isValid() const {
    return ((MUId.p1 != 0 || MUId.p2 != 0) && (MUId.instance != 0));
  }

  /// @brief Returns the file ID from the uid_t object.
  ///
  /// This method extracts the file ID from the first 64-bit field (p1) of the
  /// uid_t object by shifting it 32 bits to the right.
  ///
  /// @return A 64-bit value that represents the file ID.
  uint32_t fileId() const { return (uint32_t)(MUId.p1 >> 32); }
  /// @brief Returns the function ID from the uid_t object.
  ///
  /// This method extracts the function ID from the first 64-bit field (p1) of
  /// the uid_t object by applying a bitwise AND operation with a mask that has
  /// the lower 32 bits set to 1.
  ///
  /// @return A 64-bit value that represents the function ID.
  uint32_t functionId() const {
    return (uint32_t)(MUId.p1 & 0x00000000ffffffff);
  }
  /// @brief Returns the line number from the uid_t object.
  ///
  /// This method extracts the line number from the second 64-bit field (p2) of
  /// the uid_t object by applying a bitwise AND operation with a mask that has
  /// the lower 32 bits set to 1.
  ///
  /// @return A 32-bit value that represents the line number.
  uint32_t lineNo() { return (uint32_t)(MUId.p2 & 0x00000000ffffffff); }

  /// @brief Returns the column number from the uid_t object.
  ///
  /// This method extracts the column number from the second 64-bit field (p2)
  /// of the uid_t object by shifting it 32 bits to the right.
  ///
  /// @return A 32-bit value that represents the column number.
  uint32_t columnNo() { return (uint32_t)(MUId.p2 >> 32); }

  /// @brief Retrieves the unique identifier (UID) of the object.
  ///
  /// This method returns the complete UID of the object as an `xpti::uid128_t`
  /// structure. The UID is a composite identifier that uniquely identifies the
  /// object within the system. It may include various components such as a file
  /// ID, function ID, line number, and column number, depending on how the UID
  /// is structured in the `xpti::uid128_t` definition.
  ///
  /// @return The UID of the object as an `xpti::uid128_t` structure.
  ///
  xpti::uid128_t getUId() const { return MUId; }

  /// @brief Retrieves the instance identifier of the object.
  ///
  /// This method returns the instance identifier part of the UID, which is a
  /// unique number assigned to instances of objects. The instance identifier
  /// can be used to distinguish between different instances of objects that
  /// otherwise have the same UID components (file ID, function ID, line number,
  /// and column number). This is particularly useful in scenarios where objects
  /// are dynamically created and destroyed, and there's a need to track
  /// specific instances of these objects. The instance of an xpti::uitd_t
  /// structure is usually associated with a specific visit to a tracepoint with
  /// a payload.
  ///
  /// @return The instance identifier as a `uint64_t`.
  ///
  uint64_t getInstanceId() const { return MUId.instance; }

private:
  /// The unique identifier (UID) maintained by the helper object class.
  xpti::uid128_t MUId;
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
    // If tracing is not enabled, don't do anything
    if (!xptiTraceEnabled())
      return;

    init();
    // We expect the payload input has been populated with the information
    // available at that time; before we use this payload, we need to check if a
    // tracepoint has been set at a higher scope.
    uint64_t uid = xptiGetUniversalId();
    if (uid != xpti::invalid_uid) {
      // We already have a parent SW layer that has a tracepoint defined. This
      // should be associated with a trace event and a payload
      m_trace_event =
          const_cast<xpti::trace_event_data_t *>(xptiFindEvent(uid));
      // If the trace event is valid, extract the payload
      if (m_trace_event) {
        m_payload = m_trace_event->reserved.payload;
      } else {
        // Trace event is unavailable, so let is create one with the payload
        // associated with the UID;
        m_payload = xptiQueryPayloadByUID(uid);
        m_trace_event = xptiMakeEvent(
            m_default_name, const_cast<xpti::payload_t *>(m_payload),
            m_default_event_type, m_default_activity_type, &m_instID);
      }
    } else if (p) {
      // We may have a valid Payload
      m_top = true;
      uid = xptiRegisterPayload(p);
      // If the payload is valid, we will have a valid UID
      if (uid != xpti::invalid_uid) {
        xptiSetUniversalId(uid); // Set TLS with the UID
        m_payload = xptiQueryPayloadByUID(uid);
        m_trace_event = xptiMakeEvent(
            m_default_name, const_cast<xpti::payload_t *>(m_payload),
            m_default_event_type, m_default_activity_type, &m_instID);
      } else {
      }
    }
  }
  // Constructor that makes calls to xpti API layer to register strings and
  // create the Universal ID that is stored in the TLS entry for lookup; this
  // constructor is needed when only code location information is available
  tracepoint_t(const char *fileName, const char *funcName, int line, int column,
               void *codeptr = nullptr)
      : m_payload(nullptr), m_top(false) {
    // If tracing is not enabled, don't do anything
    if (!xptiTraceEnabled())
      return;
    init();

    // Before we use the code location information, we need to check if a
    // tracepoint has been set at a higher scope.
    uint64_t uid = xptiGetUniversalId();
    if (uid != xpti::invalid_uid) {
      // We already have a parent SW layer that has a tracepoint defined. This
      // should be associated with a trace event and a payload
      m_trace_event =
          const_cast<xpti::trace_event_data_t *>(xptiFindEvent(uid));
      // If the trace event is valid, extract the payload
      if (m_trace_event) {
        m_payload = m_trace_event->reserved.payload;
      } else {
        // Trace event is unavailable, so let is create one with the payload
        // associated with the UID;
        m_payload = xptiQueryPayloadByUID(uid);
        m_trace_event = xptiMakeEvent(
            m_default_name, const_cast<xpti::payload_t *>(m_payload),
            m_default_event_type, m_default_activity_type, &m_instID);
      }
    } else if (fileName || funcName) {
      // We expect the the file name and function name to be valid
      m_top = true;
      // Create a payload structure from the code location data
      payload_t p(funcName, fileName, line, column, codeptr);
      // Register the payload to generate the UID
      uid = xptiRegisterPayload(&p);
      if (uid != xpti::invalid_uid) {
        xptiSetUniversalId(uid);
        m_payload = xptiQueryPayloadByUID(uid);
        m_trace_event = xptiMakeEvent(
            m_default_name, const_cast<xpti::payload_t *>(m_payload),
            m_default_event_type, m_default_activity_type, &m_instID);
      }
    }
  }
  ~tracepoint_t() {
    // If tracing is not enabled, don't do anything
    if (!xptiTraceEnabled())
      return;

    if (m_top) {
      xptiSetUniversalId(xpti::invalid_uid);
    }
  }

  tracepoint_t &stream(const char *stream_name) {
    // If tracing is not enabled, don't do anything
    if (xptiTraceEnabled()) {
      m_default_stream = xptiRegisterStream(stream_name);
    }
    return *this;
  }

  tracepoint_t &trace_type(xpti::trace_point_type_t type) {
    m_default_trace_type = (uint16_t)type;
    return *this;
  }

  tracepoint_t &event_type(xpti::trace_event_type_t type) {
    if (xptiTraceEnabled()) {
      m_default_event_type = (uint16_t)type;
      if (m_trace_event)
        m_trace_event->event_type = m_default_event_type;
    }
    return *this;
  }

  tracepoint_t &activity_type(xpti::trace_activity_type_t type) {
    if (xptiTraceEnabled()) {
      m_default_activity_type = type;
      if (m_trace_event)
        m_trace_event->activity_type = (uint16_t)m_default_activity_type;
    }
    return *this;
  }

  tracepoint_t &parent_event(xpti::trace_event_data_t *event) {
    if (xptiTraceEnabled()) {
      m_parent_event = event;
    }
    return *this;
  }

  void notify(const void *user_data) {
    // If tracing is not enabled, don't notify
    if (!xptiTraceEnabled())
      return;

    xptiNotifySubscribers(m_default_stream, m_default_trace_type,
                          m_parent_event, m_trace_event, m_instID, user_data);
  }
  // The payload object that is returned will have the UID object populated and
  // can be looked up in the xpti lookup APIs or be used to make an event.
  const payload_t *payload() { return m_payload; }

  // If the tracepoint has been successfully created, the trace event will be
  // set; this method allows us to query and reuse
  const xpti::trace_event_data_t *trace_event() { return m_trace_event; }

  // Method to extract the stream used by the current tracepoint type
  uint8_t stream_id() { return m_default_stream; }

  // Method to extract the instance ID used by the current tracepoint type
  uint64_t instance_id() { return m_instID; }

  // Method to override the instance ID generated by the xptiMakeEvent() call
  void override_instance_id(uint64_t instance) { m_instID = instance; }

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
  /// @brief Initializes the default values for some parameters
  void init() {
    m_default_stream = xptiRegisterStream("diagnostics");
    m_default_trace_type = (uint16_t)xpti::trace_point_type_t::diagnostics;
    m_default_event_type = (uint16_t)xpti::trace_event_type_t::algorithm;
    m_default_activity_type = xpti::trace_activity_type_t::active;
    m_default_name = "Message"; // Likely never used
  }
  /// The payload data structure that is prepared from code_location(),
  /// caller_callee string or kernel name/codepointer based on the opt-in flag.
  const payload_t *m_payload;
  /// Indicates if the Payload was added to TLS by current instance
  bool m_top;
  /// We define a default stream to push notifications to
  uint8_t m_default_stream;
  /// We define a default trace type for the notifications which can be
  /// overridden
  uint16_t m_default_trace_type;
  /// Default sting to use in Notify() calls
  const char *m_default_name;
  /// Holds the event type that qualifies the event (as algorithm etc)
  uint16_t m_default_event_type;
  /// Holds the activity type; only needed to qualify activity
  xpti::trace_activity_type_t m_default_activity_type;
  /// Parent anc child trace event objects for graph actions
  xpti::trace_event_data_t *m_trace_event = nullptr, *m_parent_event = nullptr;
  /// Instance number of the event
  uint64_t m_instID;
};

/// @brief Checkpoint data type helps with Universal ID propagation
/// @details The class is a convenience class to support the propagation of
/// universal ID. This is a scoped class and ensures that the propagation is
/// possible withing the scope of the function that uses this service
///
/// Usage:-
/// void foo() {
/// #ifdef XPTI_TRACE_ENABLED
///   xpti::framework::checkpoint_t t(Object->uid);
/// #endif
///   ...
///   ...
/// }
///
///  See also: xptiCheckPointTest in xpti_correctness_tests.cpp
class checkpoint_t {
public:
  checkpoint_t(uint64_t universal_id) {
    // If tracing is not enabled, don't do anything
    if (!xptiTraceEnabled())
      return;

    // Let's check if TLS is currently active; if so, we will just use that
    uint64_t uid = xptiGetUniversalId();
    if (uid == xpti::invalid_uid) {
      // If the payload is valid, we will have a valid UID
      if (universal_id != xpti::invalid_uid) {
        m_top = true;
        m_uid = universal_id;
        xptiSetUniversalId(m_uid); // Set TLS with the UID
      }
    }
  }
  // Payload is queries each time and returned if the universal ID is valid
  const payload_t *payload() {
    if (m_uid != xpti::invalid_uid)
      return xptiQueryPayloadByUID(m_uid);
    else
      return nullptr;
  }

  ~checkpoint_t() {
    // If tracing is not enabled, don't do anything
    if (!xptiTraceEnabled())
      return;

    if (m_top) {
      xptiSetUniversalId(xpti::invalid_uid);
    }
  }

private:
  /// The payload data structure that is prepared from code_location(),
  /// caller_callee string or kernel name/codepointer based on the opt-in
  /// flag.
  uint64_t m_uid = xpti::invalid_uid;
  /// Indicates if the Payload was added to TLS by current instance
  bool m_top = false;
};
} // namespace framework
} // namespace xpti
