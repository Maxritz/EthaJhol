#ifndef VG_THREADPOOL_H
#define VG_THREADPOOL_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct VG_ThreadPool VG_ThreadPool;

typedef void (*VG_ThreadPoolFunc)(void *arg);

VG_ThreadPool *vg_threadpool_create(unsigned n_workers);
void vg_threadpool_destroy(VG_ThreadPool *pool);
void vg_threadpool_submit(VG_ThreadPool *pool, VG_ThreadPoolFunc func, void *arg);
/* Blocks until every submitted task (including this batch) has finished. */
void vg_threadpool_wait(VG_ThreadPool *pool);
unsigned vg_threadpool_workers(const VG_ThreadPool *pool);

/* Split [0, n_items) into worker-sized contiguous chunks and run fn(arg, begin, end) on the
 * pool. The calling thread executes the first chunk, so a single-worker pool still makes
 * progress. Returns once every chunk has completed. */
void vg_threadpool_parallel_for(VG_ThreadPool *pool,
                                void (*fn)(void *arg, uint32_t begin, uint32_t end),
                                void *arg, uint32_t n_items);

#ifdef __cplusplus
}
#endif
#endif
