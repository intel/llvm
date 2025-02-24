/*
 *
 * Copyright (C) 2022-2024 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
 * Exceptions. See LICENSE.TXT
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#include <algorithm>
#include <functional>
#include <memory>
#include <thread>
#ifndef UR_UTIL_H
#define UR_UTIL_H 1

#include <ur_api.h>

#include <atomic>
#include <iostream>
#include <map>
#include <optional>
#include <sstream>
#include <string.h>
#include <string>
#include <vector>

int ur_getpid(void);
int ur_close_fd(int fd);
int ur_duplicate_fd(int pid, int fd_in);

/* for compatibility with non-clang compilers */
#if defined(__has_feature)
#define CLANG_HAS_FEATURE(x) __has_feature(x)
#else
#define CLANG_HAS_FEATURE(x) 0
#endif

/* define for running with address sanitizer */
#if CLANG_HAS_FEATURE(address_sanitizer) || defined(__SANITIZE_ADDRESS__)
#define SANITIZER_ADDRESS
#endif

/* define for running with memory sanitizer */
#if CLANG_HAS_FEATURE(thread_sanitizer) || defined(__SANITIZE_THREAD__)
#define SANITIZER_THREAD
#endif

/* define for running with memory sanitizer */
#if CLANG_HAS_FEATURE(memory_sanitizer)
#define SANITIZER_MEMORY
#endif

/* define for running with any sanitizer runtime */
#if defined(SANITIZER_MEMORY) || defined(SANITIZER_ADDRESS) ||                 \
    defined(SANITIZER_THREAD)
#define SANITIZER_ANY
#endif
///////////////////////////////////////////////////////////////////////////////
#if defined(_WIN32)
#define MAKE_LIBRARY_NAME(NAME, VERSION) NAME ".dll"
#define STATIC_LIBRARY_EXTENSION ".lib"
#else
#if defined(__APPLE__)
#define MAKE_LIBRARY_NAME(NAME, VERSION) "lib" NAME "." VERSION ".dylib"
#else
#define MAKE_LIBRARY_NAME(NAME, VERSION) "lib" NAME ".so." VERSION
#endif
#define STATIC_LIBRARY_EXTENSION ".a"
#endif

inline std::string create_library_path(const char *name, const char *path) {
  std::string library_path;
  if (path && (strcmp("", path) != 0)) {
    library_path.assign(path);
#ifdef _WIN32
    library_path.append("\\");
#else
    library_path.append("/");
#endif
    library_path.append(name);
  } else {
    library_path.assign(name);
  }
  return library_path;
}

//////////////////////////////////////////////////////////////////////////
#if !defined(_WIN32) && (__GNUC__ >= 4)
#define __urdlllocal __attribute__((visibility("hidden")))
#else
#define __urdlllocal
#endif

///////////////////////////////////////////////////////////////////////////////
std::optional<std::string> ur_getenv(const char *name);

inline bool getenv_tobool(const char *name, bool def = false) {
  if (auto env = ur_getenv(name); env) {
    std::transform(env->begin(), env->end(), env->begin(),
                   [](unsigned char c) { return std::tolower(c); });
    auto true_str = {"y", "yes", "t", "true", "1"};
    return std::find(true_str.begin(), true_str.end(), *env) != true_str.end();
  }

  return def;
}

inline std::optional<uint64_t> getenv_to_unsigned(const char *name) try {
  auto env = ur_getenv(name);
  return env ? std::optional(std::stoi(*env)) : std::nullopt;
} catch (...) {
  return std::nullopt;
}

static void throw_wrong_format_vec(const char *env_var_name,
                                   std::string env_var_value) {
  std::stringstream ex_ss;
  ex_ss << "Wrong format of the " << env_var_name
        << " environment variable value: '" << env_var_value << "'\n"
        << "Proper format is: "
           "ENV_VAR=\"value_1\",\"value_2\",\"value_3\"";
  throw std::invalid_argument(ex_ss.str());
}

static void throw_wrong_format_map(const char *env_var_name,
                                   std::string env_var_value) {
  std::stringstream ex_ss;
  ex_ss << "Wrong format of the " << env_var_name
        << " environment variable value: '" << env_var_value << "'\n"
        << "Proper format is: "
           "ENV_VAR=\"param_1:value_1,value_2;param_2:value_1\"";
  throw std::invalid_argument(ex_ss.str());
}

/// @brief Get a vector of values from an environment variable \p env_var_name
///        A comma is a delimiter for extracting values from env var string.
///        Colons and semicolons are allowed only inside quotes to align with
///        the similar getenv_to_map() util function and avoid confusion.
///        A vector with a single value is allowed.
///        Env var must consist of strings separated by commas, ie.:
///        ENV_VAR=1,4K,2M
/// @param env_var_name name of an environment variable to be parsed
/// @return std::optional with a possible vector of strings containing parsed
/// values
///         and std::nullopt when the environment variable is not set or is
///         empty
/// @throws std::invalid_argument() when the parsed environment variable has
/// wrong format
inline std::optional<std::vector<std::string>>
getenv_to_vec(const char *env_var_name) {
  char values_delim = ',';

  auto env_var = ur_getenv(env_var_name);
  if (!env_var.has_value()) {
    return std::nullopt;
  }

  auto is_quoted = [](std::string &str) {
    return (str.front() == '\'' && str.back() == '\'') ||
           (str.front() == '"' && str.back() == '"');
  };
  auto has_colon = [](std::string &str) {
    return str.find(':') != std::string::npos;
  };
  auto has_semicolon = [](std::string &str) {
    return str.find(';') != std::string::npos;
  };

  std::vector<std::string> values_vec;
  std::stringstream ss(*env_var);
  std::string value;
  while (std::getline(ss, value, values_delim)) {
    if (value.empty() ||
        (!is_quoted(value) && (has_colon(value) || has_semicolon(value)))) {
      throw_wrong_format_vec(env_var_name, *env_var);
    }

    if (is_quoted(value)) {
      value.erase(value.cbegin());
      value.erase(value.cend() - 1);
    }

    values_vec.push_back(value);
  }

  return values_vec;
}

using EnvVarMap = std::map<std::string, std::vector<std::string>>;

/// @brief Get a map of parameters and their values from an environment variable
///        \p env_var_name
///        Semicolon is a delimiter for extracting key-values pairs from
///        an env var string. Colon is a delimiter for splitting key-values
///        pairs into keys and their values. Comma is a delimiter for values.
///        All special characters in parameter and value strings are allowed
///        except the delimiters. Env vars without parameter names are not
///        allowed, use the getenv_to_vec() util function instead. Keys in a map
///        are parsed parameters and values are vectors of strings containing
///        parameters' values, ie.:
///        ENV_VAR="param_1:value_1,value_2;param_2:value_1"
///        result map:
///             map[param_1] = [value_1, value_2]
///             map[param_2] = [value_1]
/// @param env_var_name name of an environment variable to be parsed
/// @return std::optional with a possible map with parsed parameters as keys and
///         vectors of strings containing parsed values as keys.
///         Otherwise, optional is set to std::nullopt when the environment
///         variable is not set or is empty.
/// @throws std::invalid_argument() when the parsed environment variable has
/// wrong format
inline std::optional<EnvVarMap> getenv_to_map(const char *env_var_name,
                                              bool reject_empty = true) {
  char main_delim = ';';
  char key_value_delim = ':';
  char values_delim = ',';
  EnvVarMap map;

  auto env_var = ur_getenv(env_var_name);
  if (!env_var.has_value()) {
    return std::nullopt;
  }

  auto is_quoted = [](std::string &str) {
    return (str.front() == '\'' && str.back() == '\'') ||
           (str.front() == '"' && str.back() == '"');
  };
  auto has_colon = [](std::string &str) {
    return str.find(':') != std::string::npos;
  };

  std::stringstream ss(*env_var);
  std::string key_value;
  while (std::getline(ss, key_value, main_delim)) {
    std::string key;
    std::string values;
    std::stringstream kv_ss(key_value);

    if (reject_empty && !has_colon(key_value)) {
      throw_wrong_format_map(env_var_name, *env_var);
    }

    std::getline(kv_ss, key, key_value_delim);
    std::getline(kv_ss, values);
    if (key.empty() || (reject_empty && values.empty()) ||
        map.find(key) != map.end()) {
      throw_wrong_format_map(env_var_name, *env_var);
    }

    std::vector<std::string> values_vec;
    std::stringstream values_ss(values);
    std::string value;
    while (std::getline(values_ss, value, values_delim)) {
      if (value.empty() || (has_colon(value) && !is_quoted(value))) {
        throw_wrong_format_map(env_var_name, *env_var);
      }
      if (is_quoted(value)) {
        value.erase(value.cbegin());
        value.pop_back();
      }
      values_vec.push_back(value);
    }
    map[key] = values_vec;
  }
  return map;
}

inline std::size_t combine_hashes(std::size_t seed) { return seed; }

template <typename T, typename... Args>
inline std::size_t combine_hashes(std::size_t seed, const T &v, Args... args) {
  return combine_hashes(seed ^ std::hash<T>{}(v), args...);
}

inline ur_result_t exceptionToResult(std::exception_ptr eptr) {
  try {
    if (eptr) {
      std::rethrow_exception(eptr);
    }
    return UR_RESULT_SUCCESS;
  } catch (std::bad_alloc &) {
    return UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
  } catch (const ur_result_t &e) {
    return e;
  } catch (...) {
    return UR_RESULT_ERROR_UNKNOWN;
  }
}

template <class> inline constexpr bool ur_always_false_t = false;

namespace {
// Compile-time map, mapping a UR list node type, to the enum tag type
// These are helpers for the `find_stype_node` helper below
template <ur_structure_type_t val> struct stype_map_impl {
  static constexpr ur_structure_type_t value = val;
};

template <typename T> struct stype_map {};
// contains definitions of the map specializations e.g.
// template <> struct stype_map<ur_usm_device_desc_t> :
// stype_map_impl<UR_STRUCTURE_TYPE_USM_DEVICE_DESC> {};
#include "stype_map_helpers.def"

template <typename T> constexpr int as_stype() { return stype_map<T>::value; }

/// Walk a generic UR linked list looking for a node of the given type. If it's
/// found, its address is returned, othewise `nullptr`. e.g. to find out whether
/// a `ur_usm_host_desc_t` exists in the given polymorphic list, `mylist`:
///
/// ```cpp
/// auto *node = find_stype_node<ur_usm_host_desc_t>(mylist);
/// if (!node)
///   printf("node of expected type not found!\n");
/// ```
template <typename T, typename P>
typename std::conditional_t<std::is_const_v<std::remove_pointer_t<P>>,
                            const T *, T *>
find_stype_node(P list_head) noexcept {
  auto *list = reinterpret_cast<const T *>(list_head);
  for (const auto *next = reinterpret_cast<const T *>(list); next;
       next = reinterpret_cast<const T *>(next->pNext)) {
    if (next->stype == as_stype<T>()) {
      if constexpr (!std::is_const_v<P>) {
        return const_cast<T *>(next);
      } else {
        return next;
      }
    }
  }
  return nullptr;
}
} // namespace

namespace ur {
[[noreturn]] inline void unreachable() {
#ifdef _MSC_VER
  __assume(0);
#else
  __builtin_unreachable();
#endif
}
} // namespace ur

inline std::pair<std::string, std::string>
splitMetadataName(const std::string &metadataName) {
  size_t splitPos = metadataName.rfind('@');
  if (splitPos == std::string::npos) {
    return std::make_pair(metadataName, std::string{});
  }
  return std::make_pair(metadataName.substr(0, splitPos),
                        metadataName.substr(splitPos, metadataName.length()));
}

// A simple spinlock, must be kept trivially destructible
// so that it's safe to use after its destructor has been called.
template <typename T> class Spinlock {
public:
  Spinlock() {}

  T *acquire() {
    while (lock.test_and_set(std::memory_order_acquire)) {
      std::this_thread::yield();
    }
    return &value;
  }
  void release() { lock.clear(std::memory_order_release); }

  T *bypass() { return &value; }

private:
  std::atomic_flag lock;
  T value;
};

// A reference counted pointer.
template <typename T> class Rc {
public:
  Rc() : ptr(nullptr), refcount(0) {}
  Rc(const Rc &) = delete;
  Rc &operator=(const Rc &) = delete;
  Rc(Rc &&) = delete;
  Rc &operator=(Rc &&) = delete;

  T *get_or_create(bool refcounted = true) {
    if (refcounted) {
      refcount++;
    }

    if (ptr == nullptr) {
      ptr = new T();
    }
    return ptr;
  }

  T *get_direct() { return ptr; }

  int release(std::function<void(T *)> deleter) {
    if (refcount <= 0) {
      return -1;
    }

    if (--refcount == 0) {
      deleter(ptr);
      ptr = nullptr;
    }

    return 0;
  }

  void forceDelete() {
    delete ptr;
    refcount = 0;
    ptr = nullptr;
  }

private:
  // can be read, unsyncrhonized, from multiple threads
  std::atomic<T *> ptr;
  size_t refcount;
};

// AtomicSingleton is for those cases where we want to support creating state
// on first use, global MT-safe reference-counted access, explicit synchronized
// deletion, and, on top of all that, need to gracefully handle situations where
// destructor order causes a library/application to call into the loader after
// it has been destroyed.
template <typename T> class AtomicSingleton {
private:
  static Spinlock<Rc<T>> instance;
  // Simply using an std::mutex would have been much simpler, but mutexes might
  // get deleted prior to last use of this type.

public:
  static T *get() {
    auto val = instance.acquire();

    auto ptr = val->get_or_create();

    instance.release();

    return ptr;
  }

  static T *get_direct() {
    auto ptr = instance.bypass()->get_direct();
    if (ptr == nullptr) {
      auto val = instance.acquire();
      ptr = val->get_or_create(false);
      instance.release();
    }

    // This ptr is *not* safe to access if
    // this thread is not holding a refcount
    // for this object, because some other thread
    // can come in and release it. But the alternative
    // would be to do proper refcounting on every access
    // to loader context objects. Just to secure against
    // a misbehaving application that accesses loader state
    // after it has been teardown'ed, from multiple threads.
    // Probably not worth it.

    return ptr;
  }

  static int release(std::function<void(T *)> deleter) {
    auto val = instance.acquire();
    int ret = val->release(std::move(deleter));
    instance.release();

    return ret;
  }

  // When we don't care about the refcount or the refcount is external.
  static void forceDelete() {
    auto val = instance.acquire();

    val->forceDelete();

    instance.release();
  }
};

template <typename Numeric>
static inline std::string groupDigits(Numeric numeric) {
  auto number = std::to_string(numeric);
  std::string sign = numeric >= 0 ? "" : "-";
  auto digits = number.substr(sign.size(), number.size() - sign.size());

  std::string separated;

  for (size_t i = 0; i < digits.size(); i++) {
    separated.push_back(digits[i]);

    if (i != digits.size() - 1 && (digits.size() - i - 1) % 3 == 0) {
      separated.push_back('\'');
    }
  }

  return sign + separated;
}

template <typename T> Spinlock<Rc<T>> AtomicSingleton<T>::instance;

#endif /* UR_UTIL_H */
