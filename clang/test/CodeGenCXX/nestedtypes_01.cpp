/////////////////////////////////////////////////////////////////////////////////////////
// Ensure that debug info for _M_insert_aux method is generated even if -fno-system-debug
// is used.  This is necessary for debugger access to ::value_type.
/////////////////////////////////////////////////////////////////////////////////////////

// RUN: %clang -fno-system-debug -emit-llvm -S -g %s -o %t.ll
// RUN: FileCheck --check-prefix=CHECK %s < %t.ll

// CHECK: DISubprogram(name: "_M_insert_aux",


#include <deque>
#include <iostream>

template <typename T>
class Handle
{
    T *     mPointee;
    size_t *mCount;

public:
    Handle(T *p = 0) : mPointee(p), mCount(new size_t(1)) {}
    Handle(const Handle &other) : mPointee(other.mPointee), mCount(other.mCount) { ++*mCount; }
    ~Handle() { dispose(); }

    Handle &    operator = (const Handle &other) { dispose(); mPointee = other.mPointee; mCount = other.mCount; ++mCount; }

    const T *   operator -> () const { return mPointee; }

private:
    void    dispose()
    {
        if (!--*mCount)
        {
            delete mPointee;
            delete mCount;
        }
    }
};

class Cl0
{
    int mX;

public:
    Cl0(int x) : mX(x) {}

    int getX() const { return mX; }
};

int main()
{
    std::deque<Cl0> Cl0Seq;
    std::deque<Handle<Cl0> > HCl0Seq;

    for (size_t i = 0; i < 10; ++i)
    {
        Cl0Seq.push_back(Cl0(i));
        HCl0Seq.push_back(new Cl0(i));
    }

    for (std::deque<Cl0>::iterator i = Cl0Seq.begin(); i != Cl0Seq.end(); ++i)
    {
        // Locals are not visible inside this for loop
        const Cl0 &curCl0 = *i;

        std::cout << curCl0.getX() << std::endl; // <-- Breakpoint here
    }

    for (std::deque<Handle<Cl0> >::iterator i = HCl0Seq.begin(); i != HCl0Seq.end(); ++i)
    {
        // Cl0Seq and HCl0Seq show "(error) | 0" for each element
        Handle<Cl0> curHCl0 = *i;

        std::cout << curHCl0->getX() << std::endl; // <-- Breakpoint here
    }
}
