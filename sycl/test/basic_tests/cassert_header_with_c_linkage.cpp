// RUN: %clangxx -fsycl -c %s

// Verify that compilation works when cassert is wrapped by a C linkage
// specification.

#ifdef __cplusplus
extern "C" {
#endif

#include <cassert>

#ifdef __cplusplus
}
#endif
