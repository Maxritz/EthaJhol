#ifndef VG_TRACE_H
#define VG_TRACE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct VG_TraceEvent {
    const char *name;
    uint64_t start_ns;
    uint64_t end_ns;
    uint64_t bytes;
    uint32_t category;
} VG_TraceEvent;

typedef void (*VG_TraceSink)(const VG_TraceEvent *event, void *user);
void vg_trace_set_sink(VG_TraceSink sink, void *user);
uint64_t vg_trace_now_ns(void);
void vg_trace_emit(const char *name, uint64_t start_ns, uint64_t end_ns,
                   uint64_t bytes, uint32_t category);

#ifdef __cplusplus
}
#endif
#endif
