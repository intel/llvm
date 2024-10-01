// RUN: %clangxx -fsycl -DASSERT -c %s
// RUN: %clangxx -fsycl -DCASSERT -c %s

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
