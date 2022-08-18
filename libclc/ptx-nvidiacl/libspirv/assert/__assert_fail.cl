#include <spirv/spirv.h>

void __assertfail(const char *__message, const char *__file, unsigned __line,
                  const char *__function, size_t __charSize);

_CLC_DECL void __assert_fail(const char *expr, const char *file,
                             unsigned int line, const char *func) {
  __assertfail(expr, file, line, func, 1);
}

_CLC_DECL void _wassert(const char *_Message, const char *_File,
                        unsigned _Line) {
  __assertfail(_Message, _File, _Line, 0, 1);
}
