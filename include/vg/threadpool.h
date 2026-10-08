#ifndef VG_THREADPOOL_H
#define VG_THREADPOOL_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct VG_ThreadPool VG_ThreadPool;

typedef void (*VG_ThreadPoolFunc)(void *arg);

VG_ThreadPool *vg_threadpool_create(unsigned n_workers);
void vg_threadpool_destroy(VG_ThreadPool *pool);
void vg_threadpool_submit(VG_ThreadPool *pool, VG_ThreadPoolFunc func, void *arg);
void vg_threadpool_wait(VG_ThreadPool *pool);

#ifdef __cplusplus
}
#endif
#endif
