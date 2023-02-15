#ifndef CLC_FUNC
#define CLC_FUNC

#define _CLC_OVERLOAD __attribute__((overloadable))
#define _CLC_DECL
// avoid inlines for SPIR-V related targets since we'll optimise later in the
// chain
#if defined(CLC_SPIRV) || defined(CLC_SPIRV64) || defined(CLC_CLSPV) || \
    defined(CLC_CLSPV64)
#define _CLC_DEF
#else
#define _CLC_DEF __attribute__((always_inline))
#endif
#define _CLC_INLINE __attribute__((always_inline)) inline
#define _CLC_CONVERGENT __attribute__((convergent))
#define _CLC_PURE __attribute__((pure))
#define _CLC_CONSTFN __attribute__((const))

#endif // CLC_FUNC
