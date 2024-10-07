// RUN: %clangxx -fsycl -DASSERT -fsyntax-only %s
// RUN: %clangxx -fsycl -DCASSERT -fsyntax-only %s

// Verify that compilation works when assert.h/cassert is wrapped by a C linkage
// specification.

#ifdef __cplusplus
extern "C" {
#endif

#if defined(ASSERT)
#include <assert.h>
#elif defined(CASSERT)
#include <cassert>
#endif

#ifdef __cplusplus
}
#endif
