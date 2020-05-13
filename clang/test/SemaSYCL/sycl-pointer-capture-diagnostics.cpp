// RUN: %clang_cc1  -fsycl -triple spir64 -fsycl-is-device -verify -fsyntax-only  %s
//
// Pointer variables captured by kernel lambda are checked.
// Ensure those diagnostics are working correctly.

namespace std {
class type_info;
typedef __typeof__(sizeof(int)) size_t;
} // namespace std

inline namespace cl {
namespace sycl {

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel_single_task(Func kernelFunc) {
  kernelFunc(); //#call_kernelFunc
}

typedef int device;
typedef int context;
typedef double queue;

//-- Mock USM memory allocation functions.
namespace usm {
enum class alloc { host,
                   device,
                   shared,
                   unknown };
} // namespace usm

void *malloc(std::size_t sz, const device &dev, const context &ctxt, cl::sycl::usm::alloc kind);
void *malloc(std::size_t sz, const queue &q, cl::sycl::usm::alloc kind);
void *malloc_device(std::size_t sz, const device &dev, const context &ctxt);
void *malloc_device(std::size_t sz, const queue &q);
void *malloc_shared(std::size_t sz, const device &dev, const context &ctxt);
void *malloc_shared(std::size_t sz, const queue &q);
void *malloc_host(std::size_t sz, const context &ctxt);
void *malloc_host(std::size_t sz, const queue &q);
void *aligned_alloc(std::size_t alignment, std::size_t sz, const device &dev, const context &ctxt, cl::sycl::usm::alloc kind);
void *aligned_alloc(std::size_t alignment, std::size_t sz, const queue &q, cl::sycl::usm::alloc kind);
void *aligned_alloc_device(std::size_t alignment, std::size_t sz, const device &dev, const context &ctxt);
void *aligned_alloc_device(std::size_t alignment, std::size_t sz, const queue &q);
void *aligned_alloc_shared(std::size_t alignment, std::size_t sz, const device &dev, const context &ctxt);
void *aligned_alloc_shared(std::size_t alignment, std::size_t sz, const queue &q);
void *aligned_alloc_host(std::size_t alignment, std::size_t sz, const context &ctxt);
void *aligned_alloc_host(std::size_t alignment, std::size_t sz, const queue &q);

//template form
template <typename T>
T *malloc_shared(std::size_t Count, const device &Dev, const context &Ctxt) {
  return static_cast<T *>(malloc_shared(Count * sizeof(T), Dev, Ctxt));
}

} // namespace sycl
} // namespace cl

void *malloc(std::size_t sz);
void *calloc(std::size_t num, std::size_t sz);
//-- End Mocks

struct Mesh {
  float a;
};

// User functions that might allocate memory in some way unknown to us.
float *unknownFunc();
Mesh *unknownMeshF();

void allocateUSMByHandle(float **pointerHandle, std::size_t sz, sycl::device &dev, sycl::context &ctxt) {
  float *mem = sycl::malloc_shared<float>(sz, dev, ctxt);
  *pointerHandle = mem;
}

float calledFromLambda(float *first) {
  return first[0];
}

int something(float *fromParam) {

  int device = 0, context = 0;
  double queue = 0;

  //-- Declarations

  //-- various bad pointers
  float stackFloat = 20.0;
  float *stackFloatP = &stackFloat; //#decl_stackFloatP
  float *neverInitialized;
  float *stackFloatP2 = &stackFloat;

  // std::string is already caught by 'non-trivially copy constructible' check.
  // so we only worry about literal strings.
  auto stringLiteral = "omgwtf"; //#decl_stringLiteral

  //-- various 'unknown' pointers.
  // No message or error. Up to dev to ensure they don't crash.

  //fromParam

  float *apocryphal = unknownFunc(); //#decl_apocryphal
  float *usmByHandle;
  allocateUSMByHandle(&usmByHandle, 10, device, context);

  //this one initialized as bad, but later changed. No error emitted
  float *firstBadLaterGood = &stackFloat;
  firstBadLaterGood = sycl::malloc_shared<float>(1, device, context);

  //-- structs
  Mesh stackMesh;
  stackMesh.a = 31.0;
  Mesh *stackMeshP = &stackMesh; //#decl_stackMeshP
  Mesh *neverInitializedMeshP;
  Mesh *mallocMeshP = static_cast<Mesh *>(malloc(sizeof(Mesh)));  //#decl_mallocMeshP
  Mesh *mallocMeshP2 = static_cast<Mesh *>(malloc(sizeof(Mesh))); //#decl_mallocMeshP2
  Mesh *unknownMeshP = unknownMeshF();
  Mesh *usmMeshP = static_cast<Mesh *>(sycl::malloc_shared(sizeof(Mesh), device, context)); //#decl_usmMeshP
  Mesh *usmMeshP2 = sycl::malloc_shared<Mesh>(1, device, context);                          //#decl_usmMeshP2

  //-- malloc
  float *mallocFloatP = static_cast<float *>(malloc(sizeof(float) * 2));  //#decl_mallocFloatP
  float *mallocFloatP2 = static_cast<float *>(malloc(sizeof(float) * 2)); //#decl_mallocFloatP2
  float *mallocFloatP3 = static_cast<float *>(malloc(sizeof(float) * 2)); 
  float *callocFloatP = static_cast<float *>(calloc(2, sizeof(float)));   //#decl_callocFloatP
  float *callocFloatP2 = static_cast<float *>(calloc(2, sizeof(float)));  //#decl_callocFloatP2

  //usm
  float *usmSharedP = static_cast<float *>(sycl::malloc_shared(sizeof(float), device, context));
  float *usmSharedP2 = static_cast<float *>(sycl::malloc_shared(sizeof(float), queue));
  float *usmSharedP3 = static_cast<float *>(sycl::malloc(sizeof(float), device, context, cl::sycl::usm::alloc::shared));
  float *usmSharedP4 = static_cast<float *>(sycl::malloc(sizeof(float), queue, cl::sycl::usm::alloc::shared));
  float *usmSharedP5 = sycl::malloc_shared<float>(2, device, context);

  float *usmShAlignP = static_cast<float *>(sycl::aligned_alloc_shared(1, sizeof(float), device, context));
  float *usmShAlignP2 = static_cast<float *>(sycl::aligned_alloc_shared(1, sizeof(float), queue));
  float *usmShAlignP3 = static_cast<float *>(sycl::aligned_alloc(1, sizeof(float), device, context, cl::sycl::usm::alloc::shared));
  float *usmShAlignP4 = static_cast<float *>(sycl::aligned_alloc(1, sizeof(float), queue, cl::sycl::usm::alloc::shared));

  float *usmHostP = static_cast<float *>(sycl::malloc_host(sizeof(float), context));
  float *usmHostP2 = static_cast<float *>(sycl::malloc_host(sizeof(float), queue));
  float *usmHostP3 = static_cast<float *>(sycl::malloc(sizeof(float), device, context, cl::sycl::usm::alloc::host));
  float *usmHostP4 = static_cast<float *>(sycl::malloc(sizeof(float), queue, cl::sycl::usm::alloc::host));

  float *usmHoAlignP = static_cast<float *>(sycl::aligned_alloc_host(1, sizeof(float), context));
  float *usmHoAlignP2 = static_cast<float *>(sycl::aligned_alloc_host(1, sizeof(float), queue));
  float *usmHoAlignP3 = static_cast<float *>(sycl::aligned_alloc(1, sizeof(float), device, context, cl::sycl::usm::alloc::host));
  float *usmHoAlignP4 = static_cast<float *>(sycl::aligned_alloc(1, sizeof(float), queue, cl::sycl::usm::alloc::host));

  float *usmDeviceP = static_cast<float *>(sycl::malloc_device(sizeof(float), device, context));
  float *usmDeviceP2 = static_cast<float *>(sycl::malloc_device(sizeof(float), queue));
  float *usmDeviceP3 = static_cast<float *>(sycl::malloc(sizeof(float), device, context, cl::sycl::usm::alloc::device));
  float *usmDeviceP4 = static_cast<float *>(sycl::malloc(sizeof(float), queue, cl::sycl::usm::alloc::device));

  float *okPRValueInit = usmSharedP5++;

  cl::sycl::kernel_single_task<class AName>([=]() {
    // --- Captures

    //-- various bad pointers
    //    all of these will cause crashes if not caught.

    // expected-note@#call_kernelFunc {{called by 'kernel_single_task<AName, (lambda}}
    // expected-note@#decl_mallocFloatP {{declared here}}
    // expected-error@+1 {{Illegal memory reference in SYCL device kernel. Use USM (malloc_shared, etc) instead.}}
    calledFromLambda(mallocFloatP);

    // expected-note@#decl_stackFloatP {{declared here}}
    // expected-error@+1 {{Illegal memory reference in SYCL device kernel. Use USM (malloc_shared, etc) instead.}}
    stackFloatP[0] = 30.0;

    neverInitialized[0] = 31.0; //will crash, not caught presently.

    // expected-note@#decl_stringLiteral {{declared here}}
    // expected-error@+1 {{Illegal memory reference in SYCL device kernel. Use USM (malloc_shared, etc) instead.}}
    char x = stringLiteral[0];

    //-- various 'unknown' pointers.
    // No message or error. Up to dev to ensure they don't crash.
    fromParam[0] = 70.0;

    apocryphal[0] = 71.0;

    usmByHandle[0] = 72.0;

    firstBadLaterGood[0] = 73.0;

    //-- struct pointers
    //    various bad struct pointer derefs will cause crashes if not caught.

    // expected-note@#decl_stackMeshP {{declared here}}
    // expected-error@+1 {{Illegal memory reference in SYCL device kernel. Use USM (malloc_shared, etc) instead.}}
    float smpa = stackMeshP->a;

    neverInitializedMeshP->a = 34.0; //will crash, not caught presently.

    // expected-note@#decl_mallocMeshP {{declared here}}
    // expected-error@+1 {{Illegal memory reference in SYCL device kernel. Use USM (malloc_shared, etc) instead.}}
    float mmpa = mallocMeshP->a;

    // expected-note@#decl_mallocMeshP2 {{declared here}}
    // expected-error@+1 {{Illegal memory reference in SYCL device kernel. Use USM (malloc_shared, etc) instead.}}
    mallocMeshP2->a = 45.0;

    //     unknown struct -- Nothing emitted. Up to developer to ensure it doesn't crash.
    unknownMeshP->a = 72.0;

    //     usm struct  -- OK
    float umpa = usmMeshP->a;
    usmMeshP2->a = 61.0;

    float sma = stackMesh.a; // Struct itself is copyable, so perfectly safe to capture it.

    //-- malloc
    //   all will crash if uncaught.

    // expected-note@#decl_mallocFloatP2 {{declared here}}
    // expected-error@+1 {{Illegal memory reference in SYCL device kernel. Use USM (malloc_shared, etc) instead.}}
    mallocFloatP2[0] = 80;

    // expected-note@#decl_callocFloatP {{declared here}}
    // expected-error@+1 {{Illegal memory reference in SYCL device kernel. Use USM (malloc_shared, etc) instead.}}
    callocFloatP[0] = 80;

    // expected-note@#decl_callocFloatP2 {{declared here}}
    // expected-error@+1 {{Illegal memory reference in SYCL device kernel. Use USM (malloc_shared, etc) instead.}}
    float someValue = *callocFloatP2;

    // --- Only the first capture of a pointer emits anything. So these accesses will NOT emit redundant diagnostics.
    calledFromLambda(mallocFloatP);
    stackFloatP[0] = 31.0;
    char y = stringLiteral[0];
    apocryphal[0] = 88.1;
    float smpa2 = stackMeshP->a;
    neverInitializedMeshP->a = 34.2;
    float mmpa2 = mallocMeshP->a;
    mallocMeshP2->a = 45.2;
    mallocFloatP2[0] = 81;
    callocFloatP[0] = 81;
    float someOtherValue = *callocFloatP2;

    // --- These captures all use USM, and should pass without any notes or errors.
    calledFromLambda(usmSharedP);
    umpa = usmMeshP->a;
    usmMeshP2->a = 61.0;
    usmSharedP[0] = 1;
    usmSharedP2[0] = 1;
    usmSharedP3[0] = 1;
    usmSharedP4[0] = 1;
    usmSharedP5[0] = 1;
    usmShAlignP[0] = 1;
    usmShAlignP2[0] = 1;
    usmShAlignP3[0] = 1;
    usmShAlignP4[0] = 1;
    usmHostP[0] = 1;
    usmHostP2[0] = 1;
    usmHostP3[0] = 1;
    usmHostP4[0] = 1;
    usmHoAlignP[0] = 1;
    usmHoAlignP2[0] = 1;
    usmHoAlignP3[0] = 1;
    usmHoAlignP4[0] = 1;
    usmDeviceP[0] = 1;
    usmDeviceP2[0] = 1;
    usmDeviceP3[0] = 1;
    usmDeviceP4[0] = 1;
    //
    okPRValueInit[0] = 81;
  });

  auto noProblemLambda = [=]() {
    // --- Outside a SYCL context no errors are emitted.
    mallocFloatP3[0] = -1.0;
    stackFloatP2[0] = -2.0; 
  };
  noProblemLambda();

  return 0;
}