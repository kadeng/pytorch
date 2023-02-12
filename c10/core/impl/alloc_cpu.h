#pragma once

#include <c10/macros/Macros.h>

#include <cstddef>

#ifdef __linux__
#include <sys/mman.h>
#endif

namespace c10 {

#ifdef __linux__
// since the default thp pagesize is 2MB, enable thp only
// for buffers of size 2MB or larger to avoid memory bloating
constexpr size_t gAlloc_threshold_thp = 2 * 1024 * 1024;
#endif

C10_API void* alloc_cpu(size_t nbytes);
C10_API void free_cpu(void* data);

} // namespace c10
