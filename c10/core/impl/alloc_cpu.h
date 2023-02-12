#pragma once

#include <c10/macros/Macros.h>

#include <cstddef>

#ifdef __linux__
#include <sys/mman.h>
#include <cstdlib>
#endif

namespace c10 {

#ifdef __linux__
// since the default thp pagesize is 2MB, enable thp only
// for buffers of size 2MB or larger to avoid memory bloating
constexpr size_t gAlloc_threshold_thp = 2 * 1024 * 1024;
static const char* thp_env = std::getenv("C10_THP_MEM_ALLOC_ENABLE");
const bool gIs_c10_thp_mem_alloc_enabled = std::atoi(thp_env != nullptr ? thp_env : "0");
#endif

C10_API void* alloc_cpu(size_t nbytes);
C10_API void free_cpu(void* data);

} // namespace c10
