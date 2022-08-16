//==------------ fpga_dsp_control.hpp --- SYCL FPGA DSP Control ------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace ext {
namespace intel {

enum class Preference { DSP, Softlogic, Compiler_default };
enum class Propagate { On, Off };

template <typename Function>
#ifdef __SYCL_DEVICE_ONLY__
[[intel::prefer_dsp]]
[[intel::propagate_dsp_preference]]
#endif // __SYCL_DEVICE_ONLY__
void math_prefer_dsp_propagate(Function f)
{
  f();
}

template <typename Function>
#ifdef __SYCL_DEVICE_ONLY__
[[intel::prefer_dsp]]
#endif // __SYCL_DEVICE_ONLY__
void math_prefer_dsp_no_propagate(Function f)
{
  f();
}

template <typename Function>
#ifdef __SYCL_DEVICE_ONLY__
[[intel::prefer_softlogic]]
[[intel::propagate_dsp_preference]]
#endif // __SYCL_DEVICE_ONLY__
void math_prefer_softlogic_propagate(Function f)
{
  f();
}

template <typename Function>
#ifdef __SYCL_DEVICE_ONLY__
[[intel::prefer_softlogic]]
#endif // __SYCL_DEVICE_ONLY__
void math_prefer_softlogic_no_propagate(Function f)
{
  f();
}

/// Preference: Control the hardware implementation of math operations in the
/// lambda expression.
/// -- DSP: Prefer math operations to be implemented in DSPs.
/// -- Softlogic: Prefer math operations to be implemented in soft-logic.
/// -- Compiler_default: Compiler determines the implementation based on the
/// data type and math operation.
///
/// Propagate: Determine the propagation of 'Preference' to function calls in
/// the lambda expression.
/// -- On: 'Preference' recursively applies to math operations in all function
/// calls in the lambda expression.
/// -- Off: 'Preference' applies only to math operations directly inside the
/// lambda expression. Math operations in function calls inside the lambda
/// expression are not affected by 'Preference'.
template <Preference my_preference = Preference::DSP,
          Propagate my_propagate = Propagate::On, typename Function>
void math_dsp_control(Function f) {
  if (my_preference == Preference::DSP) {
    if (my_propagate == Propagate::On) {
      math_prefer_dsp_propagate(f);
    } else {
      math_prefer_dsp_no_propagate(f);
    }
  } else if (my_preference == Preference::Softlogic) {
    if (my_propagate == Propagate::On) {
      math_prefer_softlogic_propagate(f);
    } else {
      math_prefer_softlogic_no_propagate(f);
    }
  } else { // my_preference == Preference::Compiler_default
    math_prefer_dsp_no_propagate([&]() { f(); });
  }
}

} // namespace intel
} // namespace ext
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
