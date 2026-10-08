#include "vg/trace.h"
#include <stddef.h>
#ifdef _WIN32
#include <windows.h>
#else
#include <sys/time.h>
#endif

static VG_TraceSink g_sink;
static void *g_user;

uint64_t vg_trace_now_ns(void) {
#ifdef _WIN32
    static LARGE_INTEGER freq; static int ready; LARGE_INTEGER now;
    if (!ready) { QueryPerformanceFrequency(&freq); ready = 1; }
    QueryPerformanceCounter(&now);
    return (uint64_t)((double)now.QuadPart * 1.0e9 / (double)freq.QuadPart);
#else
    struct timeval tv;
    if (gettimeofday(&tv, NULL) != 0) return 0;
    return (uint64_t)tv.tv_sec * 1000000000ull + (uint64_t)tv.tv_usec * 1000ull;
#endif
}
void vg_trace_set_sink(VG_TraceSink sink, void *user) { g_user = user; g_sink = sink; }
void vg_trace_emit(const char *name, uint64_t start_ns, uint64_t end_ns, uint64_t bytes, uint32_t category) { if (g_sink) { VG_TraceEvent e = { name, start_ns, end_ns, bytes, category }; g_sink(&e, g_user); } }
