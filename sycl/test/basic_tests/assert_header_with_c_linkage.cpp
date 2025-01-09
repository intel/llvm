// RUN: %clangxx -fsycl -fsyntax-only %s

// Verify that compilation works when assert.h is wrapped by a C linkage
// specification.

#ifdef __cplusplus
extern "C" {
#endif

#include <assert.h>

#ifdef __cplusplus
}
#endif
