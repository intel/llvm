// RUN: %clang -fsycl -c %s
// expected-no-diagnostics
#ifdef _WIN32

#ifdef __GNUC__
#error "__GNUC__ defined"
#endif

#ifdef __STDC__
#error "__STDC__ defined"
#endif
#endif
