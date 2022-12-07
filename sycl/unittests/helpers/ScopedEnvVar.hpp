//==----------------------- ScopedEnvVar.hpp -------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/common.hpp>

#include <cstdlib>
#include <functional>
#include <stdlib.h>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {

namespace unittest {
inline void set_env(const char *Name, const char *Value) {
#ifdef _WIN32
  (void)_putenv_s(Name, Value ? Value : "");
#else
  if (Value)
    (void)setenv(Name, Value, /*overwrite*/ 1);
  else
    (void)unsetenv(Name);
#endif
}

inline void unset_env(const char *Name) {
#ifdef _WIN32
  (void)_putenv_s(Name, "");
#else
  unsetenv(Name);
#endif
}

class ScopedEnvVar {
public:
  ScopedEnvVar(const char *Name, const char *Value,
               std::function<void()> ConfigReset)
      : MName(Name), MConfigReset(ConfigReset) {
    if (getenv(Name)) {
      MOriginalValue = std::string(getenv(Name));
    }
    set_env(Name, Value);
    MConfigReset();
  }

  ~ScopedEnvVar() {
    if (!MOriginalValue.empty()) {
      set_env(MName, MOriginalValue.c_str());
    } else {
      unset_env(MName);
    }
    MConfigReset();
  }

private:
  std::string MOriginalValue;
  const char *MName;
  std::function<void()> MConfigReset;
};
} // namespace unittest
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
