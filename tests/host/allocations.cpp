#include <cstdlib>
#include <new>

thread_local bool count_allocations = false;
thread_local size_t allocations = 0;

void* operator new(size_t size) {
    if (count_allocations) ++allocations;
    if (auto* ptr = std::malloc(size ? size : 1)) return ptr;
    throw std::bad_alloc();
}
void operator delete(void* ptr) noexcept { std::free(ptr); }
void operator delete(void* ptr, size_t) noexcept { std::free(ptr); }
