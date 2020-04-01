# CUDA vs OpenCL math builtin precision

## CUDA Guarantees
From [Appendix E.1 of the CUDA C Programming Guide][cuda_c_ulp]:

> This section specifies the error bounds of each function when executed on the device and also
> when executed on the host in the case where the host does not supply the function.
>
> The error bounds are generated from extensive but not exhaustive tests, so they are not
> guaranteed bounds.

In [Section 11.1.5 of the CUDA C Best Practices Guide][cuda_best_prac] on Math Libraries and
[Section 11.1.6 of the CUDA C Best Practices Guide][cuda_best_prac_precision] on Precision-related
Compiler Flags, there are mentions of the precision of math built-ins.

[cuda_best_prac]: https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#math-libraries
[cuda_best_prac_precision]: https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#precision-related-compiler-flags

## Single Precision
The following table uses the following sources:

- [Section 7.4 of the OpenCL 1.2 Specification][opencl_1.2_ulp]
- [Appendix E.1 of the CUDA C Programming Guide][cuda_c_ulp] which is referenced from the
  [CUDA Math API documentation][cuda_math_single]

In addition to the following table, the CUDA documentation also includes:

> Addition and multiplication are IEEE-compliant, so have a maximum error of 0.5 ulp.
>
> The recommended way to round a single-precision floating-point operand to an integer, with
> the result being a single-precision floating-point number is rintf(), not roundf(). The reason
> is that roundf() maps to an 8-instruction sequence on the device, whereas rintf() maps to a
> single instruction. truncf(), ceilf(), and floorf() each map to a single instruction as well.

OpenCL defines ULP (units in last place) as:

> If x is a real number that lies between two finite consecutive floating-point numbers a and b,
> without being equal to one of them, then ulp(x) = |b − a|, otherwise ulp(x) is the distance
> between the two non-equal finite floating-point numbers nearest x.  Moreover, ulp(NaN) is NaN.

Maximum error is defined in the CUDA documentation as:

>  The maximum error is stated as the absolute value of the difference in ulps between a correctly
>  rounded single-precision result and the result returned by the CUDA library function.

| OpenCL Built-in           | OpenCL Min Accuracy (ULP)          | CUDA Built-in      | CUDA Maximum Error (ULP)                                                                                      |
| ---------------           | -------------------------          | -------------      | ------------------------                                                                                      |
| `x + y`                   | Correctly rounded                  | `x + y`            | 0 ulp (IEEE-754 round-to-nearest-even)                                                                        |
| `x - y`                   | Correctly rounded                  | N/A                | N/A                                                                                                           |
| `x * y`                   | Correctly rounded                  | `x * y`            | 0 ulp (IEEE-754 round-to-nearest-even)                                                                        |
| [`1.0 / x`][`recip`]      | ≤ 2.5 ulp                          | `1.0 / x`          | 0 ulp (if compute capability ≥ 2 when compiled with `-prec-div=true`), 1 ulp (full range) otherwise           |
| [`x / y`][`divide`]       | ≤ 2.5 ulp                          | `x / y`            | 0 ulp (if compute capability ≥ 2 when compiled with `-prec-div=true`), 2 ulp (full range) otherwise           |
| [`acos`]                  | ≤ 4 ulp                            | [`acosf`]          | 3 ulp (full range)                                                                                            |
| [`acospi`][`acos`]        | ≤ 5 ulp                            | N/A                | N/A                                                                                                           |
| [`asin`]                  | ≤ 4 ulp                            | [`asinf`]          | 4 ulp (full range)                                                                                            |
| [`asinpi`][`asin`]        | ≤ 5 ulp                            | N/A                | N/A                                                                                                           |
| [`atan`]                  | ≤ 5 ulp                            | [`atanf`]          | 2 ulp (full range)                                                                                            |
| [`atan2`][`atan`]         | ≤ 6 ulp                            | [`atan2f`]         | 3 ulp (full range)                                                                                            |
| [`atanpi`][`atan`]        | ≤ 5 ulp                            | N/A                | N/A                                                                                                           |
| [`atan2pi`][`atan`]       | ≤ 6 ulp                            | N/A                | N/A                                                                                                           |
| [`acosh`][`acos`]         | ≤ 4 ulp                            | [`acoshf`]         | 4 ulp (full range)                                                                                            |
| [`asinh`][`asin`]         | ≤ 4 ulp                            | [`asinhf`]         | 3 ulp (full range)                                                                                            |
| [`atanh`][`atan`]         | ≤ 5 ulp                            | [`atanhf`]         | 3 ulp (full range)                                                                                            |
| [`cbrt`]                  | ≤ 2 ulp                            | [`cbrtf`]          | 1 ulp (full range)                                                                                            |
| [`ceil`]                  | Correctly rounded                  | [`ceilf`]          | 0 ulp (full range)                                                                                            |
| [`copysign`]              | 0 ulp                              | [`copysignf`]      | Undocumented.                                                                                                 |
| [`cos`]                   | ≤ 4 ulp                            | [`cosf`]           | 2 ulp (full range)                                                                                            |
| [`cosh`][`cos`]           | ≤ 4 ulp                            | [`coshf`]          | 2 ulp (full range)                                                                                            |
| [`cospi`][`cos`]          | ≤ 4 ulp                            | [`cospi`]          | 2 ulp (full range)                                                                                            |
| N/A                       | N/A                                | [`cyl_bessel_i0f`] | 6 ulp (full range)                                                                                            |
| N/A                       | N/A                                | [`cyl_bessel_i1f`] | 6 ulp (full range)                                                                                            |
| [`erfc`][`erf`]           | ≤ 16 ulp                           | [`erfcf`]          | 4 ulp (full range)                                                                                            |
| N/A                       | N/A                                | [`erfcinvf`]       | 2 ulp (full range)                                                                                            |
| N/A                       | N/A                                | [`erfcxf`]         | 4 ulp (full range)                                                                                            |
| N/A                       | N/A                                | [`erfinvf`]        | 2 ulp (full range)                                                                                            |
| [`erf`]                   | ≤ 16 ulp                           | [`erff`]           | 2 ulp (full range)                                                                                            |
| [`exp`]                   | ≤ 3 ulp                            | [`expf`]           | 2 ulp (full range)                                                                                            |
| [`exp2`][`exp`]           | ≤ 3 ulp                            | [`exp2f`]          | 2 ulp (full range)                                                                                            |
| [`exp10`][`exp`]          | ≤ 3 ulp                            | [`exp10f`]         | 2 ulp (full range)                                                                                            |
| [`expm1`][`exp`]          | ≤ 3 ulp                            | [`expm1f`]         | 1 ulp (full range)                                                                                            |
| [`fabs`]                  | 0 ulp                              | [`fabsf`]          | Undocumented.                                                                                                 |
| [`fdim`]                  | Correctly rounded                  | [`fdimf`]          | 0 ulp (full range)                                                                                            |
| [`floor`]                 | Correctly rounded                  | [`floorf`]         | 0 ulp (full range)                                                                                            |
| [`fma`]                   | Correctly rounded                  | [`fmaf`]           | 0 ulp (full range)                                                                                            |
| [`fmax`]                  | 0 ulp                              | [`fmaxf`]          | Undocumented.                                                                                                 |
| [`fmin`]                  | 0 ulp                              | [`fminf`]          | Undocumented.                                                                                                 |
| [`fmod`]                  | 0 ulp                              | [`fmodf`]          | 0 ulp (full range)                                                                                            |
| [`fract`]                 | Correctly rounded                  | N/A                | N/A                                                                                                           |
| [`frexp`]                 | 0 ulp                              | [`frexpf`]         | 0 ulp (full range)                                                                                            |
| [`hypot`]                 | ≤ 4 ulp                            | [`hypotf`]         | 3 ulp (full range)                                                                                            |
| [`ilogb`]                 | 0 ulp                              | [`ilogbf`]         | 0 ulp (full range)                                                                                            |
| N/A                       | N/A                                | [`j0f`]            | 9 ulp for `abs(x) < 8`, otherwise `2.2 x 10^(-6)`                                                             |
| N/A                       | N/A                                | [`j1f`]            | 9 ulp for `abs(x) < 8`, otherwise `2.2 x 10^(-6)`                                                             |
| N/A                       | N/A                                | [`jnf`]            | For `n = 128`, `2.2 x 10^(-6)`                                                                                |
| [`ldexp`]                 | Correctly rounded                  | [`ldexpf`]         | 0 ulp (full range)                                                                                            |
| N/A                       | N/A                                | [`lgammaf`]        | 6 ulp (outside interval `-10.001 ... -2.264; larger inside`)                                                  |
| [`log`]                   | ≤ 3 ulp                            | [`logf`]           | 1 ulp (full range)                                                                                            |
| [`log2`][`log`]           | ≤ 3 ulp                            | [`log2f`]          | 1 ulp (full range)                                                                                            |
| [`log10`][`log`]          | ≤ 3 ulp                            | [`log10f`]         | 2 ulp (full range)                                                                                            |
| [`log1p`][`log`]          | ≤ 2 ulp                            | [`log1pf`]         | 1 ulp (full range)                                                                                            |
| [`logb`][`log`]           | 0 ulp                              | [`logbf`]          | 0 ulp (full range)                                                                                            |
| N/A                       | N/A                                | [`lrintf`]         | 0 ulp (full range)                                                                                            |
| N/A                       | N/A                                | [`lroundf`]        | 0 ulp (full range)                                                                                            |
| N/A                       | N/A                                | [`llrintf`]        | 0 ulp (full range)                                                                                            |
| N/A                       | N/A                                | [`llroundf`]       | 0 ulp (full range)                                                                                            |
| [`mad`]                   | Any value allowed (infinite ulp)   | N/A                | N/A                                                                                                           |
| [`maxmag`][`mag`]         | 0 ulp                              | N/A                | N/A                                                                                                           |
| [`minmag`][`mag`]         | 0 ulp                              | N/A                | N/A                                                                                                           |
| [`modf`]                  | 0 ulp                              | [`modff`]          | 0 ulp (full range)                                                                                            |
| [`nan`]                   | 0 ulp                              | [`nanf`]           | Undocumented.                                                                                                 |
| N/A                       | N/A                                | [`nearbyintf`]     | 0 ulp (full range)                                                                                            |
| [`nextafter`]             | 0 ulp                              | [`nextafterf`]     | Undocumented.                                                                                                 |
| N/A                       | N/A                                | [`normf`]          | 4 ulp (full range)                                                                                            |
| N/A                       | N/A                                | [`normcdff`]       | 5 ulp (full range)                                                                                            |
| N/A                       | N/A                                | [`normcdfinvf`]    | 5 ulp (full range)                                                                                            |
| N/A                       | N/A                                | [`norm3df`]        | 3 ulp (full range)                                                                                            |
| N/A                       | N/A                                | [`norm4df`]        | 3 ulp (full range)                                                                                            |
| [`pow(x, y)`][`pow`]      | ≤ 16 ulp                           | [`powf`]           | 8 ulp (full range)                                                                                            |
| [`pown(x, y)`][`pow`]     | ≤ 16 ulp                           | N/A                | N/A                                                                                                           |
| [`powr(x, y)`][`pow`]     | ≤ 16 ulp                           | N/A                | N/A                                                                                                           |
| N/A                       | N/A                                | [`rcbrtf`]         | 1 ulp (full range)                                                                                            |
| N/A                       | N/A                                | [`rhypot`]         | 2 ulp (full range)                                                                                            |
| N/A                       | N/A                                | [`rnormf`]         | 3 ulp (full range)                                                                                            |
| N/A                       | N/A                                | [`rnorm3df`]       | 2 ulp (full range)                                                                                            |
| N/A                       | N/A                                | [`rnorm4df`]       | 2 ulp (full range)                                                                                            |
| [`remainder`]             | 0 ulp                              | [`remainderf`]     | 0 ulp (full range)                                                                                            |
| [`remquo`]                | 0 ulp                              | [`remquof`]        | 0 ulp (full range)                                                                                            |
| [`rint`]                  | Correctly rounded                  | [`rintf`]          | 0 ulp (full range)                                                                                            |
| [`rootn`]                 | ≤ 16 ulp                           | N/A                | N/A                                                                                                           |
| [`round`]                 | Correctly rounded                  | [`roundf`]         | 0 ulp (full range)                                                                                            |
| [`rsqrt`][`sqrt`]         | ≤ 2 ulp                            | [`rsqrtf`]         | 2 ulp (full range) (applies to `1 / sqrtf(x)` only when converted to `rsqrtf` by compiler)                    |
| N/A                       | N/A                                | [`scalbnf`]        | 0 ulp (full range)                                                                                            |
| N/A                       | N/A                                | [`scalblnf`]       | 0 ulp (full range)                                                                                            |
| [`sin`]                   | ≤ 4 ulp                            | [`sinf`]           | 2 ulp (full range)                                                                                            |
| [`sincos`][`sin`]         | ≤ 4 ulp for sine and cosine values | [`sincosf`]        | 2 ulp (full range)                                                                                            |
| N/A                       | N/A                                | [`sincospif`]      | 2 ulp (full range)                                                                                            |
| [`sinh`][`sin`]           | ≤ 4 ulp                            | [`sinhf`]          | 3 ulp (full range)                                                                                            |
| [`sinpi`][`sin`]          | ≤ 4 ulp                            | [`sinpif`]         | 2 ulp (full range)                                                                                            |
| [`sqrt`]                  | ≤ 3 ulp                            | [`sqrtf`]          | 0 ulp (when compiled with `-prec-sqrt=true`) otherwise 1 ulp if compute capability ≥ 5.2 and 3 ulp otherwise. |
| [`tan`]                   | ≤ 5 ulp                            | [`tanf`]           | 4 ulp (full range)                                                                                            |
| [`tanh`][`tan`]           | ≤ 5 ulp                            | [`tanhf`]          | 2 ulp (full range)                                                                                            |
| [`tanpi`][`tan`]          | ≤ 6 ulp                            | N/A                | N/A                                                                                                           |
| [`tgamma`]                | ≤ 16 ulp                           | [`tgammaf`]        | 11 ulp (full range)                                                                                           |
| [`trunc`]                 | Correctly rounded                  | [`truncf`]         | 0 ulp (full range)                                                                                            |
| N/A                       | N/A                                | [`y0f`]            | 9 ulp for `abs(x) < 8`, otherwise `2.2 x 10^(-6)`                                                             |
| N/A                       | N/A                                | [`y1f`]            | 9 ulp for `abs(x) < 8`, otherwise `2.2 x 10^(-6)`                                                             |
| N/A                       | N/A                                | [`ynf`]            | `ceil(2 + 2.5n)` for `abs(x) < n`, otherwise `2.2 x 10^(-6)`                                                  |
| N/A                       | N/A                                | [`isfinite`]       | N/A                                                                                                           |
| N/A                       | N/A                                | [`isinf`]          | N/A                                                                                                           |
| N/A                       | N/A                                | [`isnan`]          | N/A                                                                                                           |
| N/A                       | N/A                                | [`signbit`]        | N/A                                                                                                           |

OpenCL's `native_` math built-ins map to the same CUDA built-in as the equivalent non-`native_`
OpenCL built-in and the precision is implementation-defined:

  - [`native_cos`][`cos`]
  - [`native_divide`][`divide`]
  - [`native_exp`][`exp`]
  - [`native_exp2`][`exp`]
  - [`native_exp10`][`exp`]
  - [`native_log`][`log`]
  - [`native_log2`][`log`]
  - [`native_log10`][`log`]
  - [`native_powr`][`pow`]
  - [`native_recip`][`recip`]
  - [`native_rsqrt`][`sqrt`]
  - [`native_sin`][`sin`]
  - [`native_sqrt`][`sqrt`]
  - [`native_tan`][`tan`]

In [section 7.4 of the OpenCL 2.1 Specification][opencl_2.1_ulp], `mad` has a different requirement,
namely:

> Implemented either as a correctly rounded fma or as a multiply followed by an add both of which
> are correctly rounded.

Precision of SPIR-V math instructions for use in an OpenCL environment, can be
[found in this document][opencl_env_ulp].

[cuda_c_ulp]: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#standard-functions
[cuda_math_single]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__SINGLE.html#group__CUDA__MATH__SINGLE
[opencl_1.2_ulp]: https://www.khronos.org/registry/OpenCL/specs/opencl-1.2.pdf#page=319
[opencl_2.1_ulp]: https://www.khronos.org/registry/OpenCL/specs/2.2/html/OpenCL_C.html#relative-error-as-ulps
[opencl_env_ulp]: https://www.khronos.org/registry/OpenCL/specs/2.2/html/OpenCL_Env.html#relative-error-as-ulps

## Double Precision
The following table uses the following sources:

- [Section 7.4 of the OpenCL 1.2 Specification][opencl_1.2_dp_ulp]
- [Appendix E.1 of the CUDA C Programming Guide][cuda_c_ulp] which is referenced from the
  [CUDA Math API documentation][cuda_math_double]

CUDA defines maximum error in the same way as for single precision, and also includes:

> The recommended way to round a double-precision floating-point operand to an integer, with the result being a double-precision
> floating-point number is rint(), not round(). The reason is that round() maps to an 8-instruction sequence on the device,
> whereas rint() maps to a single instruction. trunc(), ceil(), and floor() each map to a single instruction as well.

Only differences from single precision are included. There are only changes to `1.0 / x`, `x / y`
and `sqrt` from OpenCL. All built-in names changed for CUDA and many precisions too.

| OpenCL Built-in       | OpenCL Min Accuracy (ULP)          | CUDA Built-in                                                                   | CUDA Maximum Error (ULP)                                       |
| ---------------       | -------------------------          | -------------                                                                   | ------------------------                                       |
| `x + y`               | Correctly rounded                  | `x + y`                                                                         | 0 ulp (IEEE-754 round-to-nearest-even)                         |
| `x - y`               | Correctly rounded                  | N/A                                                                             | N/A                                                            |
| `x * y`               | Correctly rounded                  | `x * y`                                                                         | 0 ulp (IEEE-754 round-to-nearest-even)                         |
| [`1.0 / x`][`recip`]  | Correctly rounded                  | `1.0 / x`                                                                       | 0 ulp (IEEE-754 round-to-nearest-even                          |
| [`x / y`][`divide`]   | Correctly rounded                  | `x / y`                                                                         | 0 ulp (IEEE-754 round-to-nearest-even)                         |
| [`acos`]              | ≤ 4 ulp                            | [`acos`][`acos`_cuda]                                                           | 1 ulp (full range)                                             |
| [`acospi`][`acos`]    | ≤ 5 ulp                            | N/A                                                                             | N/A                                                            |
| [`asin`]              | ≤ 4 ulp                            | [`asin`][`asin`_cuda]                                                           | 2 ulp (full range)                                             |
| [`asinpi`][`asin`]    | ≤ 5 ulp                            | N/A                                                                             | N/A                                                            |
| [`atan`]              | ≤ 5 ulp                            | [`atan`][`atan`_cuda]                                                           | 2 ulp (full range)                                             |
| [`atan2`][`atan`]     | ≤ 6 ulp                            | [`atan2`][`atan2`_cuda]                                                         | 2 ulp (full range)                                             |
| [`atanpi`][`atan`]    | ≤ 5 ulp                            | N/A                                                                             | N/A                                                            |
| [`atan2pi`][`atan`]   | ≤ 6 ulp                            | N/A                                                                             | N/A                                                            |
| [`acosh`][`acos`]     | ≤ 4 ulp                            | [`acosh`][`acosh`_cuda]                                                         | 2 ulp (full range)                                             |
| [`asinh`][`asin`]     | ≤ 4 ulp                            | [`asinh`][`asinh`_cuda]                                                         | 2 ulp (full range)                                             |
| [`atanh`][`atan`]     | ≤ 5 ulp                            | [`atanh`][`atanh`_cuda]                                                         | 2 ulp (full range)                                             |
| [`cbrt`]              | ≤ 2 ulp                            | [`cbrt`][`cbrt`_cuda]                                                           | 1 ulp (full range)                                             |
| [`ceil`]              | Correctly rounded                  | [`ceil`][`ceil`_cuda]                                                           | 0 ulp (full range)                                             |
| [`copysign`]          | 0 ulp                              | [`copysign`][`copysign`_cuda]                                                   | Undocumented.                                     |
| [`cos`]               | ≤ 4 ulp                            | [`cos`][`cos`_cuda]                                                             | 1 ulp (full range)                                             |
| [`cosh`][`cos`]       | ≤ 4 ulp                            | [`cosh`][`cosh`_cuda]                                                           | 1 ulp (full range)                                             |
| [`cospi`][`cos`]      | ≤ 4 ulp                            | [`cospi`][`cospi`_cuda]                                                         | 1 ulp (full range)                                             |
| N/A                   | N/A                                | [`cyl_bessel_i0`][`cyl_bessel_i0`_cuda]                                         | 6 ulp (full range)                                             |
| N/A                   | N/A                                | [`cyl_bessel_i1`][`cyl_bessel_i1`_cuda]                                         | 6 ulp (full range)                                             |
| [`erfc`][`erf`]       | ≤ 16 ulp                           | [`erfc`][`erfc`_cuda]                                                           | 4 ulp (full range)                                             |
| N/A                   | N/A                                | [`erfcinv`][`erfcinv`_cuda]                                                     | 6 ulp (full range)                                             |
| N/A                   | N/A                                | [`erfcx`][`erfcx`_cuda]                                                         | 3 ulp (full range)                                             |
| N/A                   | N/A                                | [`erfinv`][`erfinv`_cuda]                                                       | 5 ulp (full range)                                             |
| [`erf`]               | ≤ 16 ulp                           | [`erf`][`erf`_cuda]                                                             | 2 ulp (full range)                                             |
| [`exp`]               | ≤ 3 ulp                            | [`exp`][`exp`_cuda]                                                             | 1 ulp (full range)                                             |
| [`exp2`][`exp`]       | ≤ 3 ulp                            | [`exp2`][`exp2`_cuda]                                                           | 1 ulp (full range)                                             |
| [`exp10`][`exp`]      | ≤ 3 ulp                            | [`exp10`][`exp10`_cuda]                                                         | 1 ulp (full range)                                             |
| [`expm1`][`exp`]      | ≤ 3 ulp                            | [`expm1`][`expm1`_cuda]                                                         | 1 ulp (full range)                                             |
| [`fabs`]              | 0 ulp                              | [`fabs`][`fabs`_cuda]                                                           | Undocumented.                                     |
| [`fdim`]              | Correctly rounded                  | [`fdim`][`fdim`_cuda]                                                           | 0 ulp (full range)                                             |
| [`floor`]             | Correctly rounded                  | [`floor`][`floor`_cuda]                                                         | 0 ulp (full range)                                             |
| [`fma`]               | Correctly rounded                  | [`fma`][`fma`_cuda]                                                             | 0 ulp (IEEE-754 round-to-nearest-even)                         |
| [`fmax`]              | 0 ulp                              | [`fmax`][`fmax`_cuda]                                                           | Undocumented.                                     |
| [`fmin`]              | 0 ulp                              | [`fmin`][`fmin`_cuda]                                                           | Undocumented.                                     |
| [`fmod`]              | 0 ulp                              | [`fmod`][`fmod`_cuda]                                                           | 0 ulp (full range)                                             |
| [`fract`]             | Correctly rounded                  | N/A                                                                             | N/A                                                            |
| [`frexp`]             | 0 ulp                              | [`frexp`][`frexp`_cuda]                                                         | 0 ulp (full range)                                             |
| [`hypot`]             | ≤ 4 ulp                            | [`hypot`][`hypot`_cuda]                                                         | 2 ulp (full range)                                             |
| [`ilogb`]             | 0 ulp                              | [`ilogb`][`ilogb`_cuda]                                                         | 0 ulp (full range)                                             |
| N/A                   | N/A                                | [`j0`][`j0`_cuda]                                                               | 7 ulp for `abs(x) < 8`, otherwise `5 x 10^(-12)`               |
| N/A                   | N/A                                | [`j1`][`j1`_cuda]                                                               | 7 ulp for `abs(x) < 8`, otherwise `5 x 10^(-12)`               |
| N/A                   | N/A                                | [`jn`][`jn`_cuda]                                                               | For `n = 128`, `5 x 10^(-12)`                                  |
| [`ldexp`]             | Correctly rounded                  | [`ldexp`][`ldexp`_cuda]                                                         | 0 ulp (full range)                                             |
| N/A                   | N/A                                | [`lgamma`][`lgamma`_cuda]                                                       | 4 ulp (outside interval `-11.0001 ... -2.2637; larger inside`) |
| [`log`]               | ≤ 3 ulp                            | [`log`][`log`_cuda]                                                             | 1 ulp (full range)                                             |
| [`log2`][`log`]       | ≤ 3 ulp                            | [`log2`][`log2`_cuda]                                                           | 1 ulp (full range)                                             |
| [`log10`][`log`]      | ≤ 3 ulp                            | [`log10`][`log10`_cuda]                                                         | 1 ulp (full range)                                             |
| [`log1p`][`log`]      | ≤ 2 ulp                            | [`log1p`][`log1p`_cuda]                                                         | 1 ulp (full range)                                             |
| [`logb`][`log`]       | 0 ulp                              | [`logb`][`logb`_cuda]                                                           | 0 ulp (full range)                                             |
| N/A                   | N/A                                | [`lrint`][`lrint`_cuda]                                                         | 0 ulp (full range)                                             |
| N/A                   | N/A                                | [`lround`][`lround`_cuda]                                                       | 0 ulp (full range)                                             |
| N/A                   | N/A                                | [`llrint`][`llrint`_cuda]                                                       | 0 ulp (full range)                                             |
| N/A                   | N/A                                | [`llround`][`llround`_cuda]                                                     | 0 ulp (full range)                                             |
| [`mad`]               | Any value allowed (infinite ulp)   | N/A                                                                             | N/A                                                            |
| [`maxmag`][`mag`]     | 0 ulp                              | N/A                                                                             | N/A                                                            |
| [`minmag`][`mag`]     | 0 ulp                              | N/A                                                                             | N/A                                                            |
| [`modf`]              | 0 ulp                              | [`mod`][`mod`_cuda] (might be called `modf`, the documentation is inconsistent) | 0 ulp (full range)                                             |
| [`nan`]               | 0 ulp                              | [`nan`][`nan`_cuda]                                                             | Undocumented.                                     |
| N/A                   | N/A                                | [`nearbyint`][`nearbyint`_cuda]                                                 | 0 ulp (full range)                                             |
| [`nextafter`]         | 0 ulp                              | [`nextafter`][`nextafter`_cuda]                                                 | Undocumented.                                     |
| N/A                   | N/A                                | [`norm`][`norm`_cuda]                                                           | 3 ulp (full range)                                             |
| N/A                   | N/A                                | [`normcdf`][`normcdf`_cuda]                                                     | 5 ulp (full range)                                             |
| N/A                   | N/A                                | [`normcdfinv`][`normcdfinv`_cuda]                                               | 7 ulp (full range)                                             |
| N/A                   | N/A                                | [`norm3d`][`norm3d`_cuda]                                                       | 2 ulp (full range)                                             |
| N/A                   | N/A                                | [`norm4d`][`norm4d`_cuda]                                                       | 2 ulp (full range)                                             |
| [`pow(x, y)`][`pow`]  | ≤ 16 ulp                           | [`pow`][`pow`_cuda]                                                             | 2 ulp (full range)                                             |
| [`pown(x, y)`][`pow`] | ≤ 16 ulp                           | N/A                                                                             | N/A                                                            |
| [`powr(x, y)`][`pow`] | ≤ 16 ulp                           | N/A                                                                             | N/A                                                            |
| N/A                   | N/A                                | [`rcbrt`][`rcbrt`_cuda]                                                         | 1 ulp (full range)                                             |
| N/A                   | N/A                                | [`rhypot`][`rhypot`_cuda]                                                       | 1 ulp (full range)                                             |
| N/A                   | N/A                                | [`rnorm`][`rnorm`_cuda]                                                         | 2 ulp (full range)                                             |
| N/A                   | N/A                                | [`rnorm3d`][`rnorm3d`_cuda]                                                     | 1 ulp (full range)                                             |
| N/A                   | N/A                                | [`rnorm4d`][`rnorm4d`_cuda]                                                     | 1 ulp (full range)                                             |
| [`remainder`]         | 0 ulp                              | [`remainder`][`remainder`_cuda]                                                 | 0 ulp (full range)                                             |
| [`remquo`]            | 0 ulp                              | [`remquo`][`remquo`_cuda]                                                       | 0 ulp (full range)                                             |
| [`rint`]              | Correctly rounded                  | [`rint`][`rint`_cuda]                                                           | 0 ulp (full range)                                             |
| [`rootn`]             | ≤ 16 ulp                           | N/A                                                                             | N/A                                                            |
| [`round`]             | Correctly rounded                  | [`round`][`round`_cuda]                                                         | 0 ulp (full range)                                             |
| [`rsqrt`][`sqrt`]     | ≤ 2 ulp                            | [`rsqrt`][`rsqrt`_cuda]                                                         | 1 ulp (full range)                                             |
| N/A                   | N/A                                | [`scalbn`][`scalbn`_cuda]                                                       | 0 ulp (full range)                                             |
| N/A                   | N/A                                | [`scalbln`][`scalbln`_cuda]                                                     | 0 ulp (full range)                                             |
| [`sin`]               | ≤ 4 ulp                            | [`sin`][`sin`_cuda]                                                             | 1 ulp (full range)                                             |
| [`sincos`][`sin`]     | ≤ 4 ulp for sine and cosine values | [`sincos`][`sincos`_cuda]                                                       | 1 ulp (full range)                                             |
| N/A                   | N/A                                | [`sincospi`][`sincospi`_cuda]                                                   | 1 ulp (full range)                                             |
| [`sinh`][`sin`]       | ≤ 4 ulp                            | [`sinh`][`sinh`_cuda]                                                           | 1 ulp (full range)                                             |
| [`sinpi`][`sin`]      | ≤ 4 ulp                            | [`sinpi`][`sinpi`_cuda]                                                         | 1 ulp (full range)                                             |
| [`sqrt`]              | Correctly rounded                  | [`sqrt`][`sqrt`_cuda]                                                           | 0 ulp (IEEE-754 round-to-nearest-even)                         |
| [`tan`]               | ≤ 5 ulp                            | [`tan`][`tan`_cuda]                                                             | 2 ulp (full range)                                             |
| [`tanh`][`tan`]       | ≤ 5 ulp                            | [`tanh`][`tanh`_cuda]                                                           | 1 ulp (full range)                                             |
| [`tanpi`][`tan`]      | ≤ 6 ulp                            | N/A                                                                             | N/A                                                            |
| [`tgamma`]            | ≤ 16 ulp                           | [`tgamma`][`tgamma`_cuda]                                                       | 8 ulp (full range)                                             |
| [`trunc`]             | Correctly rounded                  | [`trunc`][`trunc`_cuda]                                                         | 0 ulp (full range)                                             |
| N/A                   | N/A                                | [`y0`][`y0`_cuda]                                                               | 7 ulp for `abs(x) < 8`, otherwise `5 x 10^(-12)`               |
| N/A                   | N/A                                | [`y1`][`y1`_cuda]                                                               | 7 ulp for `abs(x) < 8`, otherwise `5 x 10^(-12)`               |
| N/A                   | N/A                                | [`yn`][`yn`_cuda]                                                               | For `abs(x) > 1.5n`, otherwise `5 x 10^(-12)`                  |
| N/A                   | N/A                                | [`isfinite`][`isfinite`_cuda]                                                   | N/A                                                            |
| N/A                   | N/A                                | [`isinf`][`isinf`_cuda]                                                         | N/A                                                            |
| N/A                   | N/A                                | [`isnan`][`isnan`_cuda]                                                         | N/A                                                            |
| N/A                   | N/A                                | [`signbit`][`signbit`_cuda]                                                     | N/A                                                            |

[cuda_math_double]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__DOUBLE.html#group__CUDA__MATH__DOUBLE
[opencl_1.2_dp_ulp]: https://www.khronos.org/registry/OpenCL/specs/opencl-1.2.pdf#page=322

## Half Precision
The following tables uses the following sources:

- [Section 7.4 of the OpenCL 1.2 Specification][opencl_1.2_dp_ulp]
- [CUDA Math API documentation][cuda_math_half]

CUDA doesn't specify the ULP values for any of its half precision math builtins:

| OpenCL Built-in           | OpenCL Min Accuracy (ULP) | CUDA Built-in | CUDA Maximum Error (ULP)                                                            |
| ---------------           | ------------------------- | ------------- | ------------------------                                                            |
| N/A                       | N/A                       | [`_hadd`]     | Undocumented (only specifies "round-to-nearest-even mode")                          |
| N/A                       | N/A                       | [`_hadd_sat`] | Undocumented (only specifies "round-to-nearest-even mode")                          |
| N/A                       | N/A                       | [`hceil`]     | Undocumented                                                                        |
| [`half_cos`][`cos`]       | ≤ 8192 ulp                | [`hcos`]      | Undocumented (only specifies "round-to-nearest-even mode")                          |
| [`half_divide`][`divide`] | ≤ 8192 ulp                | [`_hdiv`]     | Undocumented (only specifies "round-to-nearest mode")                               |
| N/A                       | N/A                       | [`_heq`]      | Undocumented                                                                        |
| N/A                       | N/A                       | [`_hequ`]     | Undocumented                                                                        |
| [`half_exp`][`exp`]       | ≤ 8192 ulp                | [`hexp`]      | Undocumented (only specifies "round-to-nearest-even mode")                          |
| [`half_exp2`][`exp`]      | ≤ 8192 ulp                | [`hexp2`]     | Undocumented (only specifies "round-to-nearest-even mode")                          |
| [`half_exp10`][`exp`]     | ≤ 8192 ulp                | [`hexp10`]    | Undocumented (only specifies "round-to-nearest-even mode")                          |
| N/A                       | N/A                       | [`hfloor`]    | Undocumented                                                                        |
| N/A                       | N/A                       | [`_hfma`]     | Undocumented (only specifies "round-to-nearest-even mode")                          |
| N/A                       | N/A                       | [`_hfma_sat`] | Undocumented (only specifies "round-to-nearest-even mode")                          |
| N/A                       | N/A                       | [`_hge`]      | Undocumented                                                                        |
| N/A                       | N/A                       | [`_hgeu`]     | Undocumented                                                                        |
| N/A                       | N/A                       | [`_hgt`]      | Undocumented                                                                        |
| N/A                       | N/A                       | [`_hgtu`]     | Undocumented                                                                        |
| N/A                       | N/A                       | [`_hisinf`]   | Undocumented                                                                        |
| N/A                       | N/A                       | [`_hisnan`]   | Undocumented                                                                        |
| N/A                       | N/A                       | [`_hle`]      | Undocumented                                                                        |
| N/A                       | N/A                       | [`_hleu`]     | Undocumented                                                                        |
| [`half_log`][`log`]       | ≤ 8192 ulp                | [`hlog`]      | Undocumented (only specifies "round-to-nearest-even mode")                          |
| [`half_log2`][`log`]      | ≤ 8192 ulp                | [`hlog2`]     | Undocumented (only specifies "round-to-nearest-even mode")                          |
| [`half_log10`][`log`]     | ≤ 8192 ulp                | [`hlog10`]    | Undocumented (only specifies "round-to-nearest-even mode")                          |
| N/A                       | N/A                       | [`_hlt`]      | Undocumented                                                                        |
| N/A                       | N/A                       | [`_hltu`]     | Undocumented                                                                        |
| N/A                       | N/A                       | [`_hmul`]     | Undocumented (only specifies "round-to-nearest mode")                               |
| N/A                       | N/A                       | [`_hmul_sat`] | Undocumented (only specifies "round-to-nearest mode")                               |
| N/A                       | N/A                       | [`_hneg`]     | Undocumented                                                                        |
| N/A                       | N/A                       | [`_hne`]      | Undocumented                                                                        |
| N/A                       | N/A                       | [`_hneu`]     | Undocumented                                                                        |
| [`half_powr`][`pow`]      | ≤ 8192 ulp                | N/A           | N/A                                                                                 |
| [`half_recip`][`recip`]   | ≤ 8192 ulp                | [`hrcp`]      | Undocumented (only specifies "round-to-nearest-even mode")                          |
| N/A                       | N/A                       | [`hrint`]     | Undocumented (only specifies "halfway cases rounded to nearest even integer value") |
| [`half_rsqrt`][`sqrt`]    | ≤ 8192 ulp                | [`hrqsrt`]    | Undocumented (only specifies "round-to-nearest mode")                               |
| [`half_sin`][`sin`]       | ≤ 8192 ulp                | [`hsin`]      | Undocumented (only specifies "round-to-nearest-even mode")                          |
| [`half_sqrt`][`sqrt`]     | ≤ 8192 ulp                | [`hsqrt`]     | Undocumented (only specifies "round-to-nearest-even mode")                          |
| N/A                       | N/A                       | [`_hsub`]     | Undocumented (only specifies "round-to-nearest mode")                               |
| N/A                       | N/A                       | [`_hsub_sat`] | Undocumented (only specifies "round-to-nearest mode")                               |
| [`half_tan`][`tan`]       | ≤ 8192 ulp                | N/A           | N/A                                                                                 |
| N/A                       | N/A                       | [`htrunc`]    | Undocumented                                                                        |

CUDA also defines math builtins that operate on a `half2` type to which there is no OpenCL parallel:

| CUDA Built-in  | CUDA Maximum Error (ULP)                                                            |
| -------------  | ------------------------                                                            |
| [`_h2div`]     | Undocumented (only specifies "round-to-nearest mode")                               |
| [`_hadd2_sat`] | Undocumented (only specifies "round-to-nearest-even mode")                          |
| [`_hadd2`]     | Undocumented (only specifies "round-to-nearest-even mode")                          |
| [`_hbeg2`]     | Undocumented                                                                        |
| [`_hbegu2`]    | Undocumented                                                                        |
| [`_hbge2`]     | Undocumented                                                                        |
| [`_hbgeu2`]    | Undocumented                                                                        |
| [`_hbgt2`]     | Undocumented                                                                        |
| [`_hbgtu2`]    | Undocumented                                                                        |
| [`_hble2`]     | Undocumented                                                                        |
| [`_hbleu2`]    | Undocumented                                                                        |
| [`_hblt2`]     | Undocumented                                                                        |
| [`_hbltu2`]    | Undocumented                                                                        |
| [`_hbne2`]     | Undocumented                                                                        |
| [`_hbneu2`]    | Undocumented                                                                        |
| [`_heq2`]      | Undocumented                                                                        |
| [`_hequ2`]     | Undocumented                                                                        |
| [`_hfma2_sat`] | Undocumented (only specifies "round-to-nearest-even mode")                          |
| [`_hfma2`]     | Undocumented (only specifies "round-to-nearest-even mode")                          |
| [`_hge2`]      | Undocumented                                                                        |
| [`_hgeu2`]     | Undocumented                                                                        |
| [`_hgt2`]      | Undocumented                                                                        |
| [`_hgtu2`]     | Undocumented                                                                        |
| [`_hisnan2`]   | Undocumented                                                                        |
| [`_hle2`]      | Undocumented                                                                        |
| [`_hleu2`]     | Undocumented                                                                        |
| [`_hlt2`]      | Undocumented                                                                        |
| [`_hltu2`]     | Undocumented                                                                        |
| [`_hmul2_sat`] | Undocumented (only specifies "round-to-nearest-even mode")                          |
| [`_hmul2`]     | Undocumented (only specifies "round-to-nearest-even mode")                          |
| [`_hne2`]      | Undocumented                                                                        |
| [`_hneg2`]     | Undocumented                                                                        |
| [`_hneu2`]     | Undocumented                                                                        |
| [`_hsub2_sat`] | Undocumented (only specifies "round-to-nearest-even mode")                          |
| [`_hsub2`]     | Undocumented (only specifies "round-to-nearest-even mode")                          |
| [`h2ceil`]     | Undocumented                                                                        |
| [`h2cos`]      | Undocumented (only specifies "round-to-nearest-even mode")                          |
| [`h2exp10`]    | Undocumented (only specifies "round-to-nearest-even mode")                          |
| [`h2exp2`]     | Undocumented (only specifies "round-to-nearest-even mode")                          |
| [`h2exp`]      | Undocumented (only specifies "round-to-nearest mode")                               |
| [`h2floor`]    | Undocumented                                                                        |
| [`h2log10`]    | Undocumented (only specifies "round-to-nearest-even mode")                          |
| [`h2log2`]     | Undocumented (only specifies "round-to-nearest-even mode")                          |
| [`h2log`]      | Undocumented (only specifies "round-to-nearest-even mode")                          |
| [`h2rcp`]      | Undocumented (only specifies "round-to-nearest-even mode")                          |
| [`h2rint`]     | Undocumented (only specifies "halfway cases rounded to nearest even integer value") |
| [`h2rsqrt`]    | Undocumented (only specifies "round-to-nearest-even mode")                          |
| [`h2trunc`]    | Undocumented                                                                        |

Further, CUDA defines conversion and data movement functions:

| CUDA Built-in         | CUDA Maximum Error (ULP)                                   |
| -------------         | ------------------------                                   |
| [`__float22half2_rn`] | Undocumented (only specifies "round-to-nearest-even mode") |
| [`__float2half2_rn`]  | Undocumented (only specifies "round-to-nearest-even mode") |
| [`__float2half_rd`]   | Undocumented (only specifies "round-down mode")            |
| [`__float2half_rn`]   | Undocumented (only specifies "round-to-nearest-even mode") |
| [`__float2half_ru`]   | Undocumented (only specifies "round-up mode")              |
| [`__float2half_rz`]   | Undocumented (only specifies "round-towards-zero mode")    |
| [`__float2half`]      | Undocumented (only specifies "round-to-nearest-even mode") |
| [`__floats2half2_rn`] | Undocumented (only specifies "round-to-nearest-even mode") |
| [`__half22float2`]    | Undocumented                                               |
| [`__half2float`]      | Undocumented                                               |
| [`__half2half2`]      | Undocumented                                               |
| [`__half2int_rd`]     | Undocumented (only specifies "round-down mode")            |
| [`__half2int_rn`]     | Undocumented (only specifies "round-to-nearest-even mode") |
| [`__half2int_ru`]     | Undocumented (only specifies "round-up mode")              |
| [`__half2int_rz`]     | Undocumented (only specifies "round-towards-zero mode")    |
| [`__half2ll_rd`]      | Undocumented (only specifies "round-down mode")            |
| [`__half2ll_rn`]      | Undocumented (only specifies "round-to-nearest-even mode") |
| [`__half2ll_ru`]      | Undocumented (only specifies "round-up mode")              |
| [`__half2ll_rz`]      | Undocumented (only specifies "round-towards-zero mode")    |
| [`__half2short_rd`]   | Undocumented (only specifies "round-down mode")            |
| [`__half2short_rn`]   | Undocumented (only specifies "round-to-nearest-even mode") |
| [`__half2short_ru`]   | Undocumented (only specifies "round-up mode")              |
| [`__half2short_rz`]   | Undocumented (only specifies "round-towards-zero mode")    |
| [`__half2uint_rd`]    | Undocumented (only specifies "round-down mode")            |
| [`__half2uint_rn`]    | Undocumented (only specifies "round-to-nearest-even mode") |
| [`__half2uint_ru`]    | Undocumented (only specifies "round-up mode")              |
| [`__half2uint_rz`]    | Undocumented (only specifies "round-towards-zero mode")    |
| [`__half2ull_rd`]     | Undocumented (only specifies "round-down mode")            |
| [`__half2ull_rn`]     | Undocumented (only specifies "round-to-nearest-even mode") |
| [`__half2ull_ru`]     | Undocumented (only specifies "round-up mode")              |
| [`__half2ull_rz`]     | Undocumented (only specifies "round-towards-zero mode")    |
| [`__half2ushort_rd`]  | Undocumented (only specifies "round-down mode")            |
| [`__half2ushort_rn`]  | Undocumented (only specifies "round-to-nearest-even mode") |
| [`__half2ushort_ru`]  | Undocumented (only specifies "round-up mode")              |
| [`__half2ushort_rz`]  | Undocumented (only specifies "round-towards-zero mode")    |
| [`__half_as_short`]   | Undocumented                                               |
| [`__half_as_ushort`]  | Undocumented                                               |
| [`__halves2half2`]    | Undocumented                                               |
| [`__high2float`]      | Undocumented                                               |
| [`__high2half2`]      | Undocumented                                               |
| [`__high2half`]       | Undocumented                                               |
| [`__highs2half2`]     | Undocumented                                               |
| [`__int2half_rd`]     | Undocumented (only specifies "round-down mode")            |
| [`__int2half_rn`]     | Undocumented (only specifies "round-to-nearest-even mode") |
| [`__int2half_ru`]     | Undocumented (only specifies "round-up mode")              |
| [`__int2half_rz`]     | Undocumented (only specifies "round-towards-zero mode")    |
| [`__ll2half_rd`]      | Undocumented (only specifies "round-down mode")            |
| [`__ll2half_rn`]      | Undocumented (only specifies "round-to-nearest-even mode") |
| [`__ll2half_ru`]      | Undocumented (only specifies "round-up mode")              |
| [`__ll2half_rz`]      | Undocumented (only specifies "round-towards-zero mode")    |
| [`__low2float`]       | Undocumented                                               |
| [`__low2half2`]       | Undocumented                                               |
| [`__low2half`]        | Undocumented                                               |
| [`__lowhigh2highlow`] | Undocumented                                               |
| [`__lows2half2`]      | Undocumented                                               |
| [`__shfl_down_sync`]  | Undocumented                                               |
| [`__shfl_sync`]       | Undocumented                                               |
| [`__shfl_up_sync`]    | Undocumented                                               |
| [`__shfl_xor_sync`]   | Undocumented                                               |
| [`__short2half_rd`]   | Undocumented (only specifies "round-down mode")            |
| [`__short2half_rn`]   | Undocumented (only specifies "round-to-nearest-even mode") |
| [`__short2half_ru`]   | Undocumented (only specifies "round-up mode")              |
| [`__short2half_rz`]   | Undocumented (only specifies "round-towards-zero mode")    |
| [`__short_as_half`]   | Undocumented                                               |
| [`__uint2half_rd`]    | Undocumented (only specifies "round-down mode")            |
| [`__uint2half_rn`]    | Undocumented (only specifies "round-to-nearest-even mode") |
| [`__uint2half_ru`]    | Undocumented (only specifies "round-up mode")              |
| [`__uint2half_rz`]    | Undocumented (only specifies "round-towards-zero mode")    |
| [`__ull2half_rd`]     | Undocumented (only specifies "round-down mode")            |
| [`__ull2half_rn`]     | Undocumented (only specifies "round-to-nearest-even mode") |
| [`__ull2half_ru`]     | Undocumented (only specifies "round-up mode")              |
| [`__ull2half_rz`]     | Undocumented (only specifies "round-towards-zero mode")    |
| [`__ushort2half_rd`]  | Undocumented (only specifies "round-down mode")            |
| [`__ushort2half_rn`]  | Undocumented (only specifies "round-to-nearest-even mode") |
| [`__ushort2half_ru`]  | Undocumented (only specifies "round-up mode")              |
| [`__ushort2half_rz`]  | Undocumented (only specifies "round-towards-zero mode")    |
| [`__ushort_as_half`]  | Undocumented                                               |

[cuda_math_half]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__INTRINSIC__HALF.html#group__CUDA__MATH__INTRINSIC__HALF

[`acos`]: https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/acos.html
[`asin`]: https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/asin.html
[`atan`]: https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/atan.html
[`cbrt`]: https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/cbrt.html
[`ceil`]: https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/ceil.html
[`copysign`]: https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/copysign.html
[`cos`]: https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/cos.html
[`divide`]: https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/divide.html
[`erf`]: https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/erf.html
[`exp`]: https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/exp.html
[`fabs`]: https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/fabs.html
[`fdim`]: https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/fdim.html
[`floor`]: https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/floor.html
[`fma`]: https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/fma.html
[`fmax`]: https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/fmax.html
[`fmin`]: https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/fmin.html
[`fmod`]: https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/fmod.html
[`fract`]: https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/fract.html
[`frexp`]: https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/frexp.html
[`hypot`]: https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/hypot.html
[`ilogb`]: https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/ilogb.html
[`ldexp`]: https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/ldexp.html
[`log`]: https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/log.html
[`mad`]: https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/mad.html
[`mag`]: https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/mag.html
[`modf`]: https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/modf.html
[`nan`]: https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/nan.html
[`nextafter`]: https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/nextafter.html
[`pow`]: https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/pow.html
[`recip`]: https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/recip.html
[`remainder`]: https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/remainder.html
[`remquo`]: https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/remquo.html
[`rint`]: https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/rint.html
[`rootn`]: https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/rootn.html
[`round`]: https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/round.html
[`sin`]: https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/sin.html
[`sqrt`]: https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/sqrt.html
[`tan`]: https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/tan.html
[`tgamma`]: https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/tgamma.html
[`trunc`]: https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/trunc.html

[`acosf`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__SINGLE.html#group__CUDA__MATH__SINGLE_1g63d1c22538561dc228fc230d10d85dca
[`acoshf`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__SINGLE.html#group__CUDA__MATH__SINGLE_1gb0f45cada398311319b50a00ff7e826e
[`asinf`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__SINGLE.html#group__CUDA__MATH__SINGLE_1g82b2bb388724796ae8a30069abb3b386
[`asinhf`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__SINGLE.html#group__CUDA__MATH__SINGLE_1g74d4dabb94aa5c77ce31fd0ea987c083
[`atan2f`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__SINGLE.html#group__CUDA__MATH__SINGLE_1g3f0bdfc73288f9dda45e5c9be7811c9d
[`atanf`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__SINGLE.html#group__CUDA__MATH__SINGLE_1g82629bb4eec2d8c9c95b9c69188beff9
[`atanhf`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__SINGLE.html#group__CUDA__MATH__SINGLE_1g1b176d9d72adbf998b1960f830ad9dcc
[`cbrtf`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__SINGLE.html#group__CUDA__MATH__SINGLE_1g96d2384128af36ea9cb9b20d366900c7
[`ceilf`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__SINGLE.html#group__CUDA__MATH__SINGLE_1g43a6f3aa4ccdb026b038a3fe9a80f65d
[`copysignf`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__SINGLE.html#group__CUDA__MATH__SINGLE_1gf624240731f96c35e2bbf9aaa9217ad6
[`cosf`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__SINGLE.html#group__CUDA__MATH__SINGLE_1g20858ddd8f75a2c8332bdecd536057bf
[`coshf`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__SINGLE.html#group__CUDA__MATH__SINGLE_1g34a53cc088d117bc7045caa111279799
[`cospi`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__SINGLE.html#group__CUDA__MATH__SINGLE_1g6fc515121cf408a92ef611a3c6fdc5cc
[`cyl_bessel_i0f`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__SINGLE.html#group__CUDA__MATH__SINGLE_1gee787afb8a173c23b99d89239e245c59
[`cyl_bessel_i1f`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__SINGLE.html#group__CUDA__MATH__SINGLE_1g2505fc93886666a3ceec465ac5bfda1c
[`erfcf`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__SINGLE.html#group__CUDA__MATH__SINGLE_1g31faaaeab2a785191c3e0e66e030ceca
[`erfcinvf`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__SINGLE.html#group__CUDA__MATH__SINGLE_1g2bae6c7d986e0ab7e5cf685ac8b7236c
[`erfcxf`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__SINGLE.html#group__CUDA__MATH__SINGLE_1gec797649c94f21aecb8dc033a7b97353
[`erff`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__SINGLE.html#group__CUDA__MATH__SINGLE_1g3b8115ff34a107f4608152fd943dbf81
[`erfinvf`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__SINGLE.html#group__CUDA__MATH__SINGLE_1g3b8115ff34a107f4608152fd943dbf81
[`exp10f`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__SINGLE.html#group__CUDA__MATH__SINGLE_1g60f1de4fe78a907d915a52be29a799e7
[`exp2f`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__SINGLE.html#group__CUDA__MATH__SINGLE_1g3e2984de99de67ca680c9bb4f4427f81
[`expf`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__SINGLE.html#group__CUDA__MATH__SINGLE_1ge2d7656fe00f9e750c6f3bde8cc0dca6
[`expm1f`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__SINGLE.html#group__CUDA__MATH__SINGLE_1g832817212e7b0debe05d23ea37bdd748
[`fabsf`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__SINGLE.html#group__CUDA__MATH__SINGLE_1gb00f8593e1bfb1985526020fbec4e0fc
[`fdimf`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__SINGLE.html#group__CUDA__MATH__SINGLE_1g13959e5ca19c910e0d6f8e6ca5492149
[`floorf`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__SINGLE.html#group__CUDA__MATH__SINGLE_1gdbff62f4c1647b9694f35d053eff5288
[`fmaf`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__SINGLE.html#group__CUDA__MATH__SINGLE_1g5910ee832dab4f5d37118e0a6811c195
[`fmaxf`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__SINGLE.html#group__CUDA__MATH__SINGLE_1g6e7516db46be25c33fb26e203287f2a3
[`fminf`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__SINGLE.html#group__CUDA__MATH__SINGLE_1gbf48322ad520d7b12542edf990dde8c0
[`fmodf`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__SINGLE.html#group__CUDA__MATH__SINGLE_1g9255f64a2585463fea365c8273d23904
[`frexpf`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__SINGLE.html#group__CUDA__MATH__SINGLE_1g56e8cba742e2f80647903dac9c93eb37
[`hypotf`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__SINGLE.html#group__CUDA__MATH__SINGLE_1g7942dfc9161818074cfabacda7acd4c7
[`ilogbf`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__SINGLE.html#group__CUDA__MATH__SINGLE_1g4e9bcb254b97eb63abf3092233464131
[`isfinite`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__SINGLE.html#group__CUDA__MATH__SINGLE_1g57a3c8313f570282a1a7bcc78743b08e
[`isinf`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__SINGLE.html#group__CUDA__MATH__SINGLE_1g0a62e45f335a23ee64ecad3fb87a72e3
[`isnan`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__SINGLE.html#group__CUDA__MATH__SINGLE_1gf8093cd7c372f91c9837a82fd368c711
[`j0f`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__SINGLE.html#group__CUDA__MATH__SINGLE_1gba3e4bad4109f5e8509dc1925fade7ce
[`j1f`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__SINGLE.html#group__CUDA__MATH__SINGLE_1g462954bfc6ada6132f28bd7fce41334e
[`jnf`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__SINGLE.html#group__CUDA__MATH__SINGLE_1gdcd52a43c4f2d8d9148a022d6d6851dd
[`ldexpf`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__SINGLE.html#group__CUDA__MATH__SINGLE_1g7d82accff3d8e3307d61e028c19c30cd
[`lgammaf`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__SINGLE.html#group__CUDA__MATH__SINGLE_1gf7ffab2d685130195ba255e954e21130
[`llrintf`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__SINGLE.html#group__CUDA__MATH__SINGLE_1g7d4af230b5deee73fbfa9801f44f0616
[`llroundf`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__SINGLE.html#group__CUDA__MATH__SINGLE_1gf2a7fe8fb57e5b39886d776f75fdf5d6
[`log10f`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__SINGLE.html#group__CUDA__MATH__SINGLE_1gb49e218cf742a0eb08e5516dd5160585
[`log1pf`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__SINGLE.html#group__CUDA__MATH__SINGLE_1g9d53128ab5f7d6ebc4798f243481a6d7
[`log2f`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__SINGLE.html#group__CUDA__MATH__SINGLE_1gfc9ae1bd4ebb4cd9533a50f1bf486f08
[`logbf`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__SINGLE.html#group__CUDA__MATH__SINGLE_1g9a86f57d529d7000b04cb30e859a21b7
[`logf`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__SINGLE.html#group__CUDA__MATH__SINGLE_1gcdaf041c4071f63cba0e51658b89ffa4
[`lrintf`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__SINGLE.html#group__CUDA__MATH__SINGLE_1g639a876a55da8142dcd917ce6c12c27d
[`lroundf`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__SINGLE.html#group__CUDA__MATH__SINGLE_1g4d10236b2afbafda2fd85825811b84e3
[`modff`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__SINGLE.html#group__CUDA__MATH__SINGLE_1g7c49d2e467f6ca3cfc0362d84bb474ab
[`nanf`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__SINGLE.html#group__CUDA__MATH__SINGLE_1g372c640f910303dc4a7f17ce684322c5
[`nearbyintf`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__SINGLE.html#group__CUDA__MATH__SINGLE_1g53c10d923def0d85af5a2b65b1a021f0
[`nextafterf`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__SINGLE.html#group__CUDA__MATH__SINGLE_1g997fc003282f27b1c02c8a44fb4189f0
[`norm3df`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__SINGLE.html#group__CUDA__MATH__SINGLE_1g921612f74ed8a71e62d40c547cab6dcf
[`norm4df`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__SINGLE.html#group__CUDA__MATH__SINGLE_1g2334d82818e94dcac4251cd045e1e281
[`normcdff`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__SINGLE.html#group__CUDA__MATH__SINGLE_1g102ea4753919ee208c9b294e1c053cf1
[`normcdfinvf`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__SINGLE.html#group__CUDA__MATH__SINGLE_1g1c0a28ad7f7555ab16e0a1e409690174
[`normf`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__SINGLE.html#group__CUDA__MATH__SINGLE_1gb795748f3476add6c57a4af5f299965e
[`powf`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__SINGLE.html#group__CUDA__MATH__SINGLE_1gb519b517c0036b3604d602f716a919dd
[`rcbrtf`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__SINGLE.html#group__CUDA__MATH__SINGLE_1g937164a0d40347821ad16b5cb5069c92
[`remainderf`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__SINGLE.html#group__CUDA__MATH__SINGLE_1g36179ffa51305653b55c1e76f44154ff
[`remquof`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__SINGLE.html#group__CUDA__MATH__SINGLE_1ga0d8ebba46ca705859d1c7462b53118d
[`rhypot`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__SINGLE.html#group__CUDA__MATH__SINGLE_1ga53c41aebb09f501ea5e09a01145a932
[`rintf`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__SINGLE.html#group__CUDA__MATH__SINGLE_1g7791cd93108ffc6d24524f2e8635ccfd
[`rnorm3df`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__SINGLE.html#group__CUDA__MATH__SINGLE_1gf97228e858bd11e2934c26cf54a1dff6
[`rnorm4df`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__SINGLE.html#group__CUDA__MATH__SINGLE_1g66a3b53292754ba1c455fb9b30b1e40a
[`rnormf`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__SINGLE.html#group__CUDA__MATH__SINGLE_1g33482a663ef08bfc69557c20551e3d5f
[`roundf`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__SINGLE.html#group__CUDA__MATH__SINGLE_1ga1c1521079e51b4f54771b16a7f8aeea
[`rsqrtf`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__SINGLE.html#group__CUDA__MATH__SINGLE_1g5a9bc318028131cfd13d10abfae1ae13
[`scalblnf`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__SINGLE.html#group__CUDA__MATH__SINGLE_1gc94fa1e3aea5f190b7ceb47917e722be
[`scalbnf`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__SINGLE.html#group__CUDA__MATH__SINGLE_1ge5d0f588dbdbce27abe79ac3280a429f
[`signbit`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__SINGLE.html#group__CUDA__MATH__SINGLE_1gf105073ad5ef209e40942216f4ba6d8c
[`sincosf`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__SINGLE.html#group__CUDA__MATH__SINGLE_1g9456ff9df91a3874180d89a94b36fd46
[`sincospif`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__SINGLE.html#group__CUDA__MATH__SINGLE_1gab8978300988c385e0aa4b6cba44225e
[`sinf`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__SINGLE.html#group__CUDA__MATH__SINGLE_1g4677d53159664972c54bb697b9c1bace
[`sinhf`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__SINGLE.html#group__CUDA__MATH__SINGLE_1g72c262cde9f805d08492c316fc0158d9
[`sinpif`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__SINGLE.html#group__CUDA__MATH__SINGLE_1g85a985e497f4199be19462387e062ae2
[`sqrtf`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__SINGLE.html#group__CUDA__MATH__SINGLE_1gcb80df3c252b3feb3cc88f992b955a14
[`tanf`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__SINGLE.html#group__CUDA__MATH__SINGLE_1g561a1e0eab1092d294d331caf9bb93c5
[`tanhf`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__SINGLE.html#group__CUDA__MATH__SINGLE_1g7d925743801795775ca98ae83d4ba6e6
[`tgammaf`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__SINGLE.html#group__CUDA__MATH__SINGLE_1g0e556a6b5d691277e3234f4548d9ae23
[`truncf`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__SINGLE.html#group__CUDA__MATH__SINGLE_1g86499f47865e04e1ca845927f41b3322
[`y0f`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__SINGLE.html#group__CUDA__MATH__SINGLE_1g87d0270856e29b6a34038c017513f811
[`y1f`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__SINGLE.html#group__CUDA__MATH__SINGLE_1gbba94fdcb53f6a12f8bf5191697e8359
[`ynf`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__SINGLE.html#group__CUDA__MATH__SINGLE_1g383612b6d78a55003343521bca193ecd

[`acos`_cuda]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__DOUBLE.html#group__CUDA__MATH__DOUBLE_1gfb79b8e69174e322b3d5da70cd363521
[`acosh`_cuda]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__DOUBLE.html#group__CUDA__MATH__DOUBLE_1g41d6a7aee6b7e78987c1ea9633f6467a
[`asin`_cuda]:  https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__DOUBLE.html#group__CUDA__MATH__DOUBLE_1g8328d1b24f630bfc9747b57a13e66e79
[`asinh`_cuda]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__DOUBLE.html#group__CUDA__MATH__DOUBLE_1g10334b3ee5d54b6e6959102709af23ce
[`atan2`_cuda]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__DOUBLE.html#group__CUDA__MATH__DOUBLE_1gdd5ea203222910d0fba30d3bcfd6fbfe
[`atan`_cuda]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__DOUBLE.html#group__CUDA__MATH__DOUBLE_1g875675909708a2bd6d4e889df0e7791c
[`atanh`_cuda]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__DOUBLE.html#group__CUDA__MATH__DOUBLE_1ga8da8c2dc65bc77ced8e92475d423cb6
[`cbrt`_cuda]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__DOUBLE.html#group__CUDA__MATH__DOUBLE_1g86e3a3d10161a10246658ab77fac8311
[`ceil`_cuda]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__DOUBLE.html#group__CUDA__MATH__DOUBLE_1gc45db992bc2ed076e6f1edccd2d3e3d0
[`copysign`_cuda]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__DOUBLE.html#group__CUDA__MATH__DOUBLE_1ga06f087bfaf3245b3d78e30658eb9b2e
[`cos`_cuda]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__DOUBLE.html#group__CUDA__MATH__DOUBLE_1g3f1d2831497e6fa3f0072395e13a8ecf
[`cosh`_cuda]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__DOUBLE.html#group__CUDA__MATH__DOUBLE_1gcb71d08327c30ff681f47d5cefdf661f
[`cospi`_cuda]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__DOUBLE.html#group__CUDA__MATH__DOUBLE_1g0b7c24b9064401951cb1e66a23b44a4b
[`cyl_bessel_i0`_cuda]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__DOUBLE.html#group__CUDA__MATH__DOUBLE_1g1447f688cd7e242c793ff15eb0406da2
[`cyl_bessel_i1`_cuda]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__DOUBLE.html#group__CUDA__MATH__DOUBLE_1ga166717a7cb710679a45eb8f94258136
[`erf`_cuda]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__DOUBLE.html#group__CUDA__MATH__DOUBLE_1gbd196c4f3bc4260ffe99944b2400b951
[`erfc`_cuda]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__DOUBLE.html#group__CUDA__MATH__DOUBLE_1ge5fb0600e76f923d822e51b6148a9d1a
[`erfcinv`_cuda]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__DOUBLE.html#group__CUDA__MATH__DOUBLE_1g16e94306d9467be526954fdef161e4da
[`erfcx`_cuda]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__DOUBLE.html#group__CUDA__MATH__DOUBLE_1g31bd5945637fd6790091b3a0f77b9169
[`erfinv`_cuda]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__DOUBLE.html#group__CUDA__MATH__DOUBLE_1g2f624d3d5014335f087d6e33f370088f
[`exp10`_cuda]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__DOUBLE.html#group__CUDA__MATH__DOUBLE_1g9c59e13661f0e53fd46f1cfa231f5ff2
[`exp2`_cuda]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__DOUBLE.html#group__CUDA__MATH__DOUBLE_1g033d73c657d39a2ac311c0ecb0eedd4f
[`exp`_cuda]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__DOUBLE.html#group__CUDA__MATH__DOUBLE_1g15c1324292b08058007e4be047228e84
[`expm1`_cuda]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__DOUBLE.html#group__CUDA__MATH__DOUBLE_1g47772b17638c6b764d5ca5a6b8df1018
[`fabs`_cuda]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__DOUBLE.html#group__CUDA__MATH__DOUBLE_1g4f9fbe6c98f94000badf4ecf3211c128
[`fdim`_cuda]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__DOUBLE.html#group__CUDA__MATH__DOUBLE_1gbfbecf3022a22ba02e34a643158553e6
[`floor`_cuda]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__DOUBLE.html#group__CUDA__MATH__DOUBLE_1g4b7a1abc2e9e010b0e3f38bcdb2d1aa3
[`fma`_cuda]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__DOUBLE.html#group__CUDA__MATH__DOUBLE_1gff2117f6f3c4ff8a2aa4ce48a0ff2070
[`fmax`_cuda]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__DOUBLE.html#group__CUDA__MATH__DOUBLE_1g8f5b0627e6706e432728bd16cb326754
[`fmin`_cuda]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__DOUBLE.html#group__CUDA__MATH__DOUBLE_1gc970b9542e2d3e8e5d1e3ebb6a705dde
[`fmod`_cuda]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__DOUBLE.html#group__CUDA__MATH__DOUBLE_1g5e4d96de745c62d885d0a3a6bc838b86
[`frexp`_cuda]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__DOUBLE.html#group__CUDA__MATH__DOUBLE_1gf83b8e238282287d560dd12e7531e89f
[`hypot`_cuda]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__DOUBLE.html#group__CUDA__MATH__DOUBLE_1gc8fc174f8cc55bb32f1f6f12b4ff6c2e
[`ilogb`_cuda]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__DOUBLE.html#group__CUDA__MATH__DOUBLE_1g1085a209cbd5f56a4f2dbf1ba0f67be4
[`isfinite`_cuda]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__DOUBLE.html#group__CUDA__MATH__DOUBLE_1g366741a6f8e9847dd7268f4a005028ff
[`isinf`_cuda]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__DOUBLE.html#group__CUDA__MATH__DOUBLE_1gfe9aea186f33fb4f951f614ff2b53701
[`isnan`_cuda]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__DOUBLE.html#group__CUDA__MATH__DOUBLE_1g25649cf7c3d3c7a68423489532b8d459
[`j0`_cuda]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__DOUBLE.html#group__CUDA__MATH__DOUBLE_1g39cb9f4d5156e720837d77f518f2298a
[`j1`_cuda]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__DOUBLE.html#group__CUDA__MATH__DOUBLE_1g626a7fad13f7ab4e523e852e0686f6f3
[`jn`_cuda]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__DOUBLE.html#group__CUDA__MATH__DOUBLE_1gd4c381147beb88bc72ca3952602de721
[`ldexp`_cuda]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__DOUBLE.html#group__CUDA__MATH__DOUBLE_1g12ac38ace0d74cc339325e745cd281d5
[`lgamma`_cuda]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__DOUBLE.html#group__CUDA__MATH__DOUBLE_1g402aaedc732b2eabf59abc07d744ed35
[`llrint`_cuda]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__DOUBLE.html#group__CUDA__MATH__DOUBLE_1g6d2532344fe30f7f8988e031aac8e1cd
[`llround`_cuda]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__DOUBLE.html#group__CUDA__MATH__DOUBLE_1g6e401c3a6f291b874fc95b8480bcad02
[`log10`_cuda]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__DOUBLE.html#group__CUDA__MATH__DOUBLE_1g0aed82d571362c58f9486385383e7f64
[`log1p`_cuda]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__DOUBLE.html#group__CUDA__MATH__DOUBLE_1g3c680d660d75780ef53075a439211626
[`log2`_cuda]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__DOUBLE.html#group__CUDA__MATH__DOUBLE_1gc15d49c9960470b4791eafa0607ca777
[`log`_cuda]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__DOUBLE.html#group__CUDA__MATH__DOUBLE_1g28ce8e15ef5149c271eba95663becba2
[`logb`_cuda]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__DOUBLE.html#group__CUDA__MATH__DOUBLE_1g24e6d5c7904a61d50055d27ffe6d8fdb
[`lrint`_cuda]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__DOUBLE.html#group__CUDA__MATH__DOUBLE_1g353f5748b7addbae162dd679abf829fe
[`lround`_cuda]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__DOUBLE.html#group__CUDA__MATH__DOUBLE_1g9fdb5ef303c94dc5c428dbdb534ed1fd
[`mod`_cuda]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__DOUBLE.html#group__CUDA__MATH__DOUBLE_1gf66b786e19d90c6c519ce7b80afa97bf
[`nan`_cuda]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__DOUBLE.html#group__CUDA__MATH__DOUBLE_1g6df5511321a5ac0dfe22389b728a8a9f
[`nearbyint`_cuda]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__DOUBLE.html#group__CUDA__MATH__DOUBLE_1g2316a104cfda8362208d52238181fbfb
[`nextafter`_cuda]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__DOUBLE.html#group__CUDA__MATH__DOUBLE_1gf46b3ad97567ae96f7148a10537c8f5a
[`norm3d`_cuda]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__DOUBLE.html#group__CUDA__MATH__DOUBLE_1g0f1beab2ceb43c190bbdd53073481a87
[`norm4d`_cuda]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__DOUBLE.html#group__CUDA__MATH__DOUBLE_1g22d61aa6b93f5943c4d35a3545aace18
[`norm`_cuda]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__DOUBLE.html#group__CUDA__MATH__DOUBLE_1g7c5ebbdd1d0300094d9e34fbe5218a75
[`normcdf`_cuda]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__DOUBLE.html#group__CUDA__MATH__DOUBLE_1g8368e3ba7981942344d0be3b5d817e3f
[`normcdfinv`_cuda]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__DOUBLE.html#group__CUDA__MATH__DOUBLE_1g78e93df6c3fbade8628d33e11fc94595
[`pow`_cuda]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__DOUBLE.html#group__CUDA__MATH__DOUBLE_1g6d36757715384dc18e0483aa1f04f6c7
[`rcbrt`_cuda]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__DOUBLE.html#group__CUDA__MATH__DOUBLE_1g3f5dd3f9b81f73c644d82754986ccce6
[`remainder`_cuda]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__DOUBLE.html#group__CUDA__MATH__DOUBLE_1g852e83c233f09c146c492bfd752e0dd2
[`remquo`_cuda]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__DOUBLE.html#group__CUDA__MATH__DOUBLE_1g4235a6814bb94b3faaf73a324210c58d
[`rhypot`_cuda]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__DOUBLE.html#group__CUDA__MATH__DOUBLE_1gf1dfb4d01feaa01b0b1ff15cf57ebbc3
[`rint`_cuda]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__DOUBLE.html#group__CUDA__MATH__DOUBLE_1g3b8026edb2f2e441669845f0f3fa3bf7
[`rnorm3d`_cuda]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__DOUBLE.html#group__CUDA__MATH__DOUBLE_1g1ac4eff7fecc1121d5dcfdebc3314e80
[`rnorm4d`_cuda]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__DOUBLE.html#group__CUDA__MATH__DOUBLE_1g039d37d2d8d44f074e057489a439a758
[`rnorm`_cuda]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__DOUBLE.html#group__CUDA__MATH__DOUBLE_1g3d2150666773f15337b09aa7e1662e59
[`round`_cuda]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__DOUBLE.html#group__CUDA__MATH__DOUBLE_1gbefba28ee84ef32c44d417cfd4f615d4
[`rsqrt`_cuda]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__DOUBLE.html#group__CUDA__MATH__DOUBLE_1gf799c5cd74e63236a4a08296cb12ccbc
[`scalbln`_cuda]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__DOUBLE.html#group__CUDA__MATH__DOUBLE_1g7c931cea8bc2cfe694a6170379e5914f
[`scalbn`_cuda]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__DOUBLE.html#group__CUDA__MATH__DOUBLE_1g4923bed52b438e5bfbf574bb8ce26542
[`signbit`_cuda]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__DOUBLE.html#group__CUDA__MATH__DOUBLE_1g2bd7d6942a8b25ae518636dab9ad78a7
[`sin`_cuda]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__DOUBLE.html#group__CUDA__MATH__DOUBLE_1g3ebbca20a2937d1fe51329402880df85
[`sincos`_cuda]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__DOUBLE.html#group__CUDA__MATH__DOUBLE_1gbe0e6a063a8f38850b0323933cf3320b
[`sincospi`_cuda]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__DOUBLE.html#group__CUDA__MATH__DOUBLE_1gfc99d7acfc1b14dcb6f6db56147d2560
[`sinh`_cuda]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__DOUBLE.html#group__CUDA__MATH__DOUBLE_1gabc5c0e23e1550a6cc936baa9d65a61a
[`sinpi`_cuda]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__DOUBLE.html#group__CUDA__MATH__DOUBLE_1g06ae86e791c45c081184e605f984e733
[`sqrt`_cuda]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__DOUBLE.html#group__CUDA__MATH__DOUBLE_1g1c6fe34b4ac091e40eceeb0bae58459f
[`tan`_cuda]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__DOUBLE.html#group__CUDA__MATH__DOUBLE_1g17d00b521d79b4a4404cc593839f0b7b
[`tanh`_cuda]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__DOUBLE.html#group__CUDA__MATH__DOUBLE_1gdf7b9660a2c53c91664263d39b09242d
[`tgamma`_cuda]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__DOUBLE.html#group__CUDA__MATH__DOUBLE_1gecfb49e21fc767c952827d42268c0d48
[`trunc`_cuda]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__DOUBLE.html#group__CUDA__MATH__DOUBLE_1gaa2c1b49a1f4aa25f8ce49236089f2a8
[`y0`_cuda]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__DOUBLE.html#group__CUDA__MATH__DOUBLE_1g7eab7eb6999bde9057f22e36e7db95d4
[`y1`_cuda]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__DOUBLE.html#group__CUDA__MATH__DOUBLE_1g2560f5508d3aaec918ed7e94e96a6180
[`yn`_cuda]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__DOUBLE.html#group__CUDA__MATH__DOUBLE_1g01b473912d10252607be1870b1b2660d

[`__float22half2_rn`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF__MISC.html#group__CUDA__MATH____HALF__MISC_1gc7bebc35ea0a149ccc35f214e623424c
[`__float2half2_rn`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF__MISC.html#group__CUDA__MATH____HALF__MISC_1ge40813c17ab4b0779764e2e5e3014019
[`__float2half_rd`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF__MISC.html#group__CUDA__MATH____HALF__MISC_1g6b62243ec8796e0112a8934fe8588eda
[`__float2half_rn`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF__MISC.html#group__CUDA__MATH____HALF__MISC_1g049db0958db14ed58903a33cad7c7ad7
[`__float2half_ru`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF__MISC.html#group__CUDA__MATH____HALF__MISC_1gac96fd60f5f1363392f6b00ce7784a44
[`__float2half_rz`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF__MISC.html#group__CUDA__MATH____HALF__MISC_1gba9ddf251d3baf915f0551a1f3e96e3a
[`__float2half`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF__MISC.html#group__CUDA__MATH____HALF__MISC_1g9f330c6a82c3c502821d7a104bfbfae1
[`__floats2half2_rn`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF__MISC.html#group__CUDA__MATH____HALF__MISC_1ge367f0481e6d0fcbfe9db86a7c068e1f
[`__half22float2`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF__MISC.html#group__CUDA__MATH____HALF__MISC_1g7085e030996b689b4e2ae1868b375d62
[`__half2float`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF__MISC.html#group__CUDA__MATH____HALF__MISC_1g0b79d92cb1fd7012b9c4416e9f4a03ba
[`__half2half2`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF__MISC.html#group__CUDA__MATH____HALF__MISC_1g1108041a06791eebda5b9420958e8251
[`__half2int_rd`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF__MISC.html#group__CUDA__MATH____HALF__MISC_1g0b59a74ea4a816e0668f60b125fd53c3
[`__half2int_rn`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF__MISC.html#group__CUDA__MATH____HALF__MISC_1g9990fefa4627c2be489803af0dd153db<Paste>
[`__half2int_ru`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF__MISC.html#group__CUDA__MATH____HALF__MISC_1g4aa3e81bedaf19a38d38e32e02152fa8
[`__half2int_rz`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF__MISC.html#group__CUDA__MATH____HALF__MISC_1gd89cc9e3dc6762a7106bd46af2704c8a
[`__half2ll_rd`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF__MISC.html#group__CUDA__MATH____HALF__MISC_1g3342000665ca5b362d495a29ad772d3d
[`__half2ll_rn`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF__MISC.html#group__CUDA__MATH____HALF__MISC_1g607cc45ffefd1dc8a7acd699c9ff6778
[`__half2ll_ru`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF__MISC.html#group__CUDA__MATH____HALF__MISC_1g69a67c6a1187a491c3657d9a2b8dfb7f
[`__half2ll_rz`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF__MISC.html#group__CUDA__MATH____HALF__MISC_1g22af1c3583f0fe531c9c2bac198f958a
[`__half2short_rd`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF__MISC.html#group__CUDA__MATH____HALF__MISC_1g43249b10b57a20ae627f06791751e8f3
[`__half2short_rn`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF__MISC.html#group__CUDA__MATH____HALF__MISC_1g622d02cea8661f10dba90394987be0d3
[`__half2short_ru`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF__MISC.html#group__CUDA__MATH____HALF__MISC_1g9ac82dd9c2a7ffb28c9ef0dbc63b0986
[`__half2short_rz`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF__MISC.html#group__CUDA__MATH____HALF__MISC_1g01c1522399c61a1884badce9918764fb
[`__half2uint_rd`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF__MISC.html#group__CUDA__MATH____HALF__MISC_1g17cc53632a7c303ee064211d9ff27785
[`__half2uint_rn`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF__MISC.html#group__CUDA__MATH____HALF__MISC_1gf4b2699513866302b8ba358ebe03f6e6
[`__half2uint_ru`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF__MISC.html#group__CUDA__MATH____HALF__MISC_1g6b0061b873b6ee3917291bffa447baaa
[`__half2uint_rz`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF__MISC.html#group__CUDA__MATH____HALF__MISC_1g27bf37ee90b08f461fa3c845377600cb
[`__half2ull_rd`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF__MISC.html#group__CUDA__MATH____HALF__MISC_1g0303b752ed9086fa5c42394a6eccf68c
[`__half2ull_rn`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF__MISC.html#group__CUDA__MATH____HALF__MISC_1g65dc4d227472a030a9d5576aae9ffc88
[`__half2ull_ru`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF__MISC.html#group__CUDA__MATH____HALF__MISC_1g3d76260695a82df122826e7b148e3593
[`__half2ull_rz`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF__MISC.html#group__CUDA__MATH____HALF__MISC_1g717f454f19181aba6f33665e6053bb41
[`__half2ushort_rd`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF__MISC.html#group__CUDA__MATH____HALF__MISC_1g2e71fc128fd1084b78ae5fe856634fea
[`__half2ushort_rn`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF__MISC.html#group__CUDA__MATH____HALF__MISC_1g50e9b150b33e88bbb28f0d0002d4d0ba
[`__half2ushort_ru`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF__MISC.html#group__CUDA__MATH____HALF__MISC_1g55debed624e5f810a714496256707a41
[`__half2ushort_rz`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF__MISC.html#group__CUDA__MATH____HALF__MISC_1g16a8e266bd631105911346617c21709f
[`__half_as_short`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF__MISC.html#group__CUDA__MATH____HALF__MISC_1g9f1cd8abf8672af71947f634898b0007
[`__half_as_ushort`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF__MISC.html#group__CUDA__MATH____HALF__MISC_1g3e1130448cea6166bbfcf0426ab8ad25
[`__halves2half2`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF__MISC.html#group__CUDA__MATH____HALF__MISC_1g8a0b6b624b5e2e49d3f447e3602b511b
[`__high2float`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF__MISC.html#group__CUDA__MATH____HALF__MISC_1g91418df384ec5de88b6c6b8f95a9ecb1
[`__high2half2`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF__MISC.html#group__CUDA__MATH____HALF__MISC_1ga76abcaa154c87ac2d3270d1223252eb
[`__high2half`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF__MISC.html#group__CUDA__MATH____HALF__MISC_1gff189c4a2f52a0506ade9390b50fd275
[`__highs2half2`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF__MISC.html#group__CUDA__MATH____HALF__MISC_1g5b466bd0dc874ad53116bda6a40ea8f4
[`__int2half_rd`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF__MISC.html#group__CUDA__MATH____HALF__MISC_1g45e240c370a285ebba394ee42b42a3e2
[`__int2half_rn`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF__MISC.html#group__CUDA__MATH____HALF__MISC_1g20d9b7f0c37194d23189abd7ca17e3aa
[`__int2half_ru`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF__MISC.html#group__CUDA__MATH____HALF__MISC_1gc0125412fcf6cddfdbba64b8bed31160
[`__int2half_rz`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF__MISC.html#group__CUDA__MATH____HALF__MISC_1g6dcf24a4fe2dc10ed8d7bf6630677187
[`__ll2half_rd`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF__MISC.html#group__CUDA__MATH____HALF__MISC_1g96f0c7ee50d76b598c2da75c2c0ec462
[`__ll2half_rn`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF__MISC.html#group__CUDA__MATH____HALF__MISC_1g4e2f48947ca2e50fbab6cb75aa5b9135
[`__ll2half_ru`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF__MISC.html#group__CUDA__MATH____HALF__MISC_1gec1e52441454d2ec29c75f66ea9cf3a1
[`__ll2half_rz`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF__MISC.html#group__CUDA__MATH____HALF__MISC_1g2db342c689d6838f6ff27cfb6d0cc84e
[`__low2float`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF__MISC.html#group__CUDA__MATH____HALF__MISC_1g7f66f7c36268ee9e7881e28fcebf45e7
[`__low2half2`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF__MISC.html#group__CUDA__MATH____HALF__MISC_1g84111b2921fc2387eae11b84b506fdd3
[`__low2half`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF__MISC.html#group__CUDA__MATH____HALF__MISC_1g9e7e2d8c5fb3adca2607fca0b338b40d
[`__lowhigh2highlow`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF__MISC.html#group__CUDA__MATH____HALF__MISC_1g6f71a09819e7114c541826277572261b
[`__lows2half2`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF__MISC.html#group__CUDA__MATH____HALF__MISC_1g79158e54445b181020c51a24549b0878
[`__shfl_down_sync`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF__MISC.html#group__CUDA__MATH____HALF__MISC_1g0706091cb1b0251b584d19fcd670ae9a
[`__shfl_sync`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF__MISC.html#group__CUDA__MATH____HALF__MISC_1g553d2684b619cbd06aa9dc79f8327fcf
[`__shfl_up_sync`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF__MISC.html#group__CUDA__MATH____HALF__MISC_1g30bfac09acf5d336b462bedddabc4e2a
[`__shfl_xor_sync`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF__MISC.html#group__CUDA__MATH____HALF__MISC_1g615dc3411541ca85e1390b28a4465ff4
[`__short2half_rd`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF__MISC.html#group__CUDA__MATH____HALF__MISC_1gd4537ca10b6805efddee32741edadc82
[`__short2half_rn`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF__MISC.html#group__CUDA__MATH____HALF__MISC_1g2765cbe749db434d2ea857aaf39823ba
[`__short2half_ru`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF__MISC.html#group__CUDA__MATH____HALF__MISC_1g4c30e044018c67ab6324a1db52629804
[`__short2half_rz`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF__MISC.html#group__CUDA__MATH____HALF__MISC_1g1ae9a50d9f06818790fe042028cfa3d1
[`__short_as_half`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF__MISC.html#group__CUDA__MATH____HALF__MISC_1g9270a5a7b3972f17665261112d9afb46
[`__uint2half_rd`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF__MISC.html#group__CUDA__MATH____HALF__MISC_1geed2366d494fec6b5f6b9ceeb3c07695
[`__uint2half_rn`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF__MISC.html#group__CUDA__MATH____HALF__MISC_1gb335881e80595cb421c5ad70fd834700
[`__uint2half_ru`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF__MISC.html#group__CUDA__MATH____HALF__MISC_1g6e3bd9d9dc4c8ac396b10ff942ace3ed
[`__uint2half_rz`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF__MISC.html#group__CUDA__MATH____HALF__MISC_1gdc77f9c47b0ad82cfa94e1a4503bc5dc
[`__ull2half_rd`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF__MISC.html#group__CUDA__MATH____HALF__MISC_1gc5ee93161072343d34b56ce05e7bec03
[`__ull2half_rn`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF__MISC.html#group__CUDA__MATH____HALF__MISC_1g71c18efc764c1633c1c4de389ed971b5
[`__ull2half_ru`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF__MISC.html#group__CUDA__MATH____HALF__MISC_1g4b8ecebe04abd7e3f91b4856f428d02f
[`__ull2half_rz`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF__MISC.html#group__CUDA__MATH____HALF__MISC_1g69c0b32cafad2c2e22a566b5abfd4c65
[`__ushort2half_rd`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF__MISC.html#group__CUDA__MATH____HALF__MISC_1g777e7e20097d7f0f836319ba6db20b35
[`__ushort2half_rn`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF__MISC.html#group__CUDA__MATH____HALF__MISC_1g699899689cb0471baafa9637b30cd5f8
[`__ushort2half_ru`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF__MISC.html#group__CUDA__MATH____HALF__MISC_1gd1c6fc4ce83bd519ef985711b9d6597c
[`__ushort2half_rz`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF__MISC.html#group__CUDA__MATH____HALF__MISC_1g9957e935aca60c68680a3ce0138cd955
[`__ushort_as_half`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF__MISC.html#group__CUDA__MATH____HALF__MISC_1g0a9ecce42ad9e1947f02fe068bba82aa
[`_h2div`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF__ARITHMETIC.html#group__CUDA__MATH____HALF__ARITHMETIC_1gd4eebe93064215ca566c8606697d4c5f
[`_hadd2_sat`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF2__ARITHMETIC.html#group__CUDA__MATH____HALF2__ARITHMETIC_1g0538a877f86451df528c353c6e1156bb
[`_hadd2`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF2__ARITHMETIC.html#group__CUDA__MATH____HALF2__ARITHMETIC_1g1ed66b23eb6467bf3640c81df7af6131
[`_hadd_sat`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF__ARITHMETIC.html#group__CUDA__MATH____HALF__ARITHMETIC_1g84a949d2a10e1543ec8256f5b3fd65aa
[`_hadd`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF__ARITHMETIC.html#group__CUDA__MATH____HALF__ARITHMETIC_1ga07e44376f11eaa3865163c63372475d
[`_hbeg2`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF2__COMPARISON.html#group__CUDA__MATH____HALF2__COMPARISON_1gd0e8e130e1b25bace01ac5dacf0e76d6
[`_hbegu2`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF2__COMPARISON.html#group__CUDA__MATH____HALF2__COMPARISON_1gacb80c066faa12abffbf6d9239b92eb4
[`_hbge2`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF2__COMPARISON.html#group__CUDA__MATH____HALF2__COMPARISON_1g047fef218f7b2a2b10dbe36fe333efcb
[`_hbgeu2`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF2__COMPARISON.html#group__CUDA__MATH____HALF2__COMPARISON_1g7045f77a395b2982bd7d56061a40ffe6
[`_hbgt2`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF2__COMPARISON.html#group__CUDA__MATH____HALF2__COMPARISON_1g3c0ea9543029389bf9cb5fa743c56631
[`_hbgtu2`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF2__COMPARISON.html#group__CUDA__MATH____HALF2__COMPARISON_1gc0ee2b64b525942ae0dcf7c3e155a6ff
[`_hble2`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF2__COMPARISON.html#group__CUDA__MATH____HALF2__COMPARISON_1g74b822f6bfa6892e6763a607b24f4ef4
[`_hbleu2`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF2__COMPARISON.html#group__CUDA__MATH____HALF2__COMPARISON_1g5b04fd3513ff247a6b00985449490187
[`_hblt2`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF2__COMPARISON.html#group__CUDA__MATH____HALF2__COMPARISON_1gb978931b9e238d3c5dc79c06b2115060
[`_hbltu2`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF2__COMPARISON.html#group__CUDA__MATH____HALF2__COMPARISON_1gfa7c17beed940f96776fc102c2edd5c0
[`_hbne2`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF2__COMPARISON.html#group__CUDA__MATH____HALF2__COMPARISON_1gc6fd5b3d7d5e7cabfd4d46494599144a
[`_hbneu2`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF2__COMPARISON.html#group__CUDA__MATH____HALF2__COMPARISON_1gc7d88b855df0ea1b55cd557c2d1b7178
[`_hdiv`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF__ARITHMETIC.html#group__CUDA__MATH____HALF__ARITHMETIC_1g1e8990a950a37220731255d4d0c390c4
[`_heq2`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF2__COMPARISON.html#group__CUDA__MATH____HALF2__COMPARISON_1g30369a3a8989b09f3d3b516721127650
[`_heq`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF__COMPARISON.html#group__CUDA__MATH____HALF__COMPARISON_1g7ba3285c3ded6c6f0dbf3f2a8b3f7a6d
[`_hequ2`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF2__COMPARISON.html#group__CUDA__MATH____HALF2__COMPARISON_1g9dd11e89e74d08178d72cb296f9ff0b2
[`_hequ`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF__COMPARISON.html#group__CUDA__MATH____HALF__COMPARISON_1g752064442de1e5b1e962676a4a7baaaf
[`_hfma2_sat`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF2__ARITHMETIC.html#group__CUDA__MATH____HALF2__ARITHMETIC_1g7e8b3d4633a37543bbb6cc9010f47d36
[`_hfma2`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF2__ARITHMETIC.html#group__CUDA__MATH____HALF2__ARITHMETIC_1g43628ba21ded8b1e188a367348008dab
[`_hfma_sat`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF__ARITHMETIC.html#group__CUDA__MATH____HALF__ARITHMETIC_1g096f8ab8715837bf96457d1aedc513dc
[`_hfma`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF__ARITHMETIC.html#group__CUDA__MATH____HALF__ARITHMETIC_1gaec96bd410157b5813c940ee320175f2
[`_hge2`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF2__COMPARISON.html#group__CUDA__MATH____HALF2__COMPARISON_1gcebacfee79f6a4c17d77fd6fff3b9b31
[`_hge`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF__COMPARISON.html#group__CUDA__MATH____HALF__COMPARISON_1g5eda60bbcffc3f4c9af4a98008a249bf
[`_hgeu2`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF2__COMPARISON.html#group__CUDA__MATH____HALF2__COMPARISON_1gac67d2ad282e8de0243a215d8d576646
[`_hgeu`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF__COMPARISON.html#group__CUDA__MATH____HALF__COMPARISON_1g208f8bd81fed536fdcee0303cb716286
[`_hgt2`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF2__COMPARISON.html#group__CUDA__MATH____HALF2__COMPARISON_1gf62360cbc3cb48077823cc19a9d2dd69
[`_hgt`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF__COMPARISON.html#group__CUDA__MATH____HALF__COMPARISON_1g386dae810e042f11d3f53c9fe3455a03
[`_hgtu2`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF2__COMPARISON.html#group__CUDA__MATH____HALF2__COMPARISON_1g340b34a4ae48ceb7986d88613ba4724d
[`_hgtu`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF__COMPARISON.html#group__CUDA__MATH____HALF__COMPARISON_1g00a5e7671e731e6e2d4b85fd4051a5d0
[`_hisinf`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF__COMPARISON.html#group__CUDA__MATH____HALF__COMPARISON_1gebed49bb20d04e0391e3ef960d5e8c2d
[`_hisnan2`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF2__COMPARISON.html#group__CUDA__MATH____HALF2__COMPARISON_1gde996dfcc2b08c0f511fb3ab2f02bbba
[`_hisnan`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF__COMPARISON.html#group__CUDA__MATH____HALF__COMPARISON_1g761b5a610cb54883b6a945a12cda8fe5
[`_hle2`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF2__COMPARISON.html#group__CUDA__MATH____HALF2__COMPARISON_1g83936be3b479cf8013602f350b426b03
[`_hle`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF__COMPARISON.html#group__CUDA__MATH____HALF__COMPARISON_1gfd4af36b3c5d482b54d137d6d670a792
[`_hleu2`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF2__COMPARISON.html#group__CUDA__MATH____HALF2__COMPARISON_1ga07741f51ed23685b2faaf0339973fdb
[`_hleu`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF__COMPARISON.html#group__CUDA__MATH____HALF__COMPARISON_1g81aa929767ee526b9d8040a15327bbaf
[`_hlt2`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF2__COMPARISON.html#group__CUDA__MATH____HALF2__COMPARISON_1g63a2f5044efb987fca294254f18d2595
[`_hlt`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF__COMPARISON.html#group__CUDA__MATH____HALF__COMPARISON_1g660a4376ef2071f837655adb22c337bb
[`_hltu2`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF2__COMPARISON.html#group__CUDA__MATH____HALF2__COMPARISON_1g23bda06d273dbe605add9bdfa10d55c1
[`_hltu`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF__COMPARISON.html#group__CUDA__MATH____HALF__COMPARISON_1g610c041e3815c5ddf12e6eba614963af
[`_hmul2_sat`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF2__ARITHMETIC.html#group__CUDA__MATH____HALF2__ARITHMETIC_1g03ba1312a1e9d01fdd0db37799bef670
[`_hmul2`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF2__ARITHMETIC.html#group__CUDA__MATH____HALF2__ARITHMETIC_1gccece3396cadfbaa18883a1d28ba44b4
[`_hmul_sat`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF__ARITHMETIC.html#group__CUDA__MATH____HALF__ARITHMETIC_1g5dcde50fe0cdb1f3cc9f4b409fa370a3
[`_hmul`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF__ARITHMETIC.html#group__CUDA__MATH____HALF__ARITHMETIC_1gf2f3e02bb1d1c9992c3fe709ec826e24
[`_hne2`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF2__COMPARISON.html#group__CUDA__MATH____HALF2__COMPARISON_1g3d44c4528ede67dac29486a1d4d222fb
[`_hne`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF__COMPARISON.html#group__CUDA__MATH____HALF__COMPARISON_1g4720d765d3a0a742292e567e9768d992
[`_hneg2`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF2__ARITHMETIC.html#group__CUDA__MATH____HALF2__ARITHMETIC_1g67c6596ad65a8d9525909ad19a1fec4f
[`_hneg`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF__ARITHMETIC.html#group__CUDA__MATH____HALF__ARITHMETIC_1g50cef1b840dce4b95fd739d436d0d031
[`_hneu2`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF2__COMPARISON.html#group__CUDA__MATH____HALF2__COMPARISON_1g24e2ed9191eb9660079dc86aca28ae50
[`_hneu`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF__COMPARISON.html#group__CUDA__MATH____HALF__COMPARISON_1gb72024638614a0a906cc47963cae53ee
[`_hsub2_sat`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF2__ARITHMETIC.html#group__CUDA__MATH____HALF2__ARITHMETIC_1g678acfc121db91143d3b5f355ab3bd95
[`_hsub2`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF2__ARITHMETIC.html#group__CUDA__MATH____HALF2__ARITHMETIC_1g83b37be9530a2438665257cf0324d15b
[`_hsub_sat`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF__ARITHMETIC.html#group__CUDA__MATH____HALF__ARITHMETIC_1gcfb630a04db4e817e3be53411d7b7375
[`_hsub`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF__ARITHMETIC.html#group__CUDA__MATH____HALF__ARITHMETIC_1g966908fa24410fddec6e50d00546e57b
[`h2ceil`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF2__FUNCTIONS.html#group__CUDA__MATH____HALF2__FUNCTIONS_1gc033c574f2f8a17d5f5c05988f3c824c
[`h2cos`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF2__FUNCTIONS.html#group__CUDA__MATH____HALF2__FUNCTIONS_1g64a7a1877fc3861d2c562d41ae21a556
[`h2exp10`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF2__FUNCTIONS.html#group__CUDA__MATH____HALF2__FUNCTIONS_1gf44a54bebd8c8b2429f8e3d032265134
[`h2exp2`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF2__FUNCTIONS.html#group__CUDA__MATH____HALF2__FUNCTIONS_1gc5cda143ba8404d8fba64a4271ef2d60
[`h2exp`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF2__FUNCTIONS.html#group__CUDA__MATH____HALF2__FUNCTIONS_1gbce59641ef4b50b6b5d66bca2d6e73e8
[`h2floor`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF2__FUNCTIONS.html#group__CUDA__MATH____HALF2__FUNCTIONS_1g6f84d537d7f2ded1e010d95d4626e423
[`h2log10`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF2__FUNCTIONS.html#group__CUDA__MATH____HALF2__FUNCTIONS_1g7601f13b0f6fc9a6ec462d5141d4cd43
[`h2log2`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF2__FUNCTIONS.html#group__CUDA__MATH____HALF2__FUNCTIONS_1gc94f387ebd0fe47c5d72778d86dfc960
[`h2log`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF2__FUNCTIONS.html#group__CUDA__MATH____HALF2__FUNCTIONS_1g9fd129881966428ec0c085aae866edda
[`h2rcp`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF2__FUNCTIONS.html#group__CUDA__MATH____HALF2__FUNCTIONS_1gef1ded9d8910ab16ceb0ebf1890b691e
[`h2rint`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF2__FUNCTIONS.html#group__CUDA__MATH____HALF2__FUNCTIONS_1g8dc6d2883feda53980a92beebc41cb2f
[`h2rsqrt`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF2__FUNCTIONS.html#group__CUDA__MATH____HALF2__FUNCTIONS_1g950dce1b4afa766797614491f935ef3d
[`h2trunc`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF2__FUNCTIONS.html#group__CUDA__MATH____HALF2__FUNCTIONS_1g46015025f00169486b7d67ee98a12fe2
[`hceil`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF__FUNCTIONS.html#group__CUDA__MATH____HALF__FUNCTIONS_1g71645e62825165483767fb959ade5b75
[`hcos`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF__FUNCTIONS.html#group__CUDA__MATH____HALF__FUNCTIONS_1ga65dce71ebc0dd7d12d0834e0ab6b253
[`hexp10`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF__FUNCTIONS.html#group__CUDA__MATH____HALF__FUNCTIONS_1g9795592d7a0b36eb25ed2c57b89c5020
[`hexp2`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF__FUNCTIONS.html#group__CUDA__MATH____HALF__FUNCTIONS_1g715e831f5588ef02ef2ee6a94cb07013
[`hexp`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF__FUNCTIONS.html#group__CUDA__MATH____HALF__FUNCTIONS_1g2a3dc15a7d48a5a0dee8b12bc875e522
[`hfloor`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF__FUNCTIONS.html#group__CUDA__MATH____HALF__FUNCTIONS_1g5302f4e70c2918f6737d3c159335d681
[`hlog10`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF__FUNCTIONS.html#group__CUDA__MATH____HALF__FUNCTIONS_1g5a41dfac808cbd159c1c4ea4b738c0ae
[`hlog2`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF__FUNCTIONS.html#group__CUDA__MATH____HALF__FUNCTIONS_1g3d788d8a6fdf25890f769c147056e8b4
[`hlog`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF__FUNCTIONS.html#group__CUDA__MATH____HALF__FUNCTIONS_1g74f361f9c89fe0430d18cf1136c3a799
[`hrcp`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF__FUNCTIONS.html#group__CUDA__MATH____HALF__FUNCTIONS_1g3d221a53cabf43e2457ad8ddba3a1278
[`hrint`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF__FUNCTIONS.html#group__CUDA__MATH____HALF__FUNCTIONS_1gbbf7a989130edcbdbfbb4730f61c79b1
[`hrqsrt`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF__FUNCTIONS.html#group__CUDA__MATH____HALF__FUNCTIONS_1g57710803b15f471625469a3f43b82970
[`hsin`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF__FUNCTIONS.html#group__CUDA__MATH____HALF__FUNCTIONS_1g648019bc27fc250f350f90dc688f8430
[`hsqrt`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF__FUNCTIONS.html#group__CUDA__MATH____HALF__FUNCTIONS_1g67b9bbe48e510b6dc1c666bf34aa99a6
[`htrunc`]: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF__FUNCTIONS.html#group__CUDA__MATH____HALF__FUNCTIONS_1gee5be0d01b1f9a44a56aa2110eab5047
