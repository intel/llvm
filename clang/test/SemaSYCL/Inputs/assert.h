#ifndef MOCK_ASSERT_H
#define MOCK_ASSERT_H

#define _CRT_WIDE_(s) L##s
#define _CRT_WIDE(s) _CRT_WIDE_(s)

// Simplified version of assert.h as distributed with Windows ucrt. `assert`
// expands to `_wassert` that is decorated with `sycl_device` attribute.

#ifdef __cplusplus
extern "C" {
#endif

#undef assert

#ifdef NDEBUG

#define assert(expression) ((void)0)

#else

__attribute__((sycl_device)) void __cdecl _wassert(
    wchar_t const *_Message,
    wchar_t const *_File,
    unsigned _Line);

#define assert(expression) (void)((!!(expression)) || \
                                  (_wassert(_CRT_WIDE(#expression), _CRT_WIDE(__FILE__), (unsigned)(__LINE__)), 0))

#endif

#ifdef __cplusplus
}
#endif

#endif // MOCK_ASSERT_H
