#include <spirv/spirv.h>

void __assertfail(const char *__message, const char *__file, unsigned __line,
                  const char *__function);

_CLC_DECL void __assert_fail(const char *expr, const char *file,
                             unsigned int line, const char *func) {
  __assertfail(expr, file, line, func);
}
