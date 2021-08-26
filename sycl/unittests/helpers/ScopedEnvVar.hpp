//==----------------------- ScopedEnvVar.hpp -------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <functional>
#include <stdlib.h>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl::unittest {
inline void set_env(const char *name, const char *value) {
#ifdef _WIN32
  (void)_putenv_s(name, value);
#else
  (void)setenv(name, value, /*overwrite*/ 1);
#endif
}

inline void unset_env(const char *name) {
#ifdef _WIN32
  (void)_putenv_s(name, "");
#else
  unsetenv(name);
#endif
}

class ScopedEnvVar {
public:
  ScopedEnvVar(const char *name, const char *value,
               std::function<void()> configReset)
      : mName(name), mConfigReset(configReset) {
    set_env(name, value);
    mConfigReset();
  }

  ~ScopedEnvVar() {
    unset_env(mName);
    mConfigReset();
  }

private:
  const char *mName;
  std::function<void()> mConfigReset;
};
} // namespace sycl::unittest
} // __SYCL_INLINE_NAMESPACE(cl)
