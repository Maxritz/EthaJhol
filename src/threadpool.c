#include "vg/threadpool.h"

#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#include <windows.h>
#else
#include <pthread.h>
#endif

typedef struct VG_Task {
    VG_ThreadPoolFunc func;
    void *arg;
    struct VG_Task *next;
} VG_Task;

struct VG_ThreadPool {
    VG_Task *head;
    VG_Task *tail;
#ifdef _WIN32
    CRITICAL_SECTION lock;
    HANDLE *threads;
    HANDLE semaphore;
    CONDITION_VARIABLE done_cond;
#else
    pthread_mutex_t lock;
    pthread_cond_t cond;
    pthread_cond_t done_cond;
    pthread_t *threads;
#endif
    unsigned n_workers;
    unsigned n_pending;
    int shutdown;
};

/* n_pending is decremented *after* the task body finishes so that vg_threadpool_wait()
 * never returns while a worker is still computing. */
#ifdef _WIN32
static DWORD WINAPI worker_thread(LPVOID param) {
    VG_ThreadPool *pool = (VG_ThreadPool *)param;
    for (;;) {
        WaitForSingleObject(pool->semaphore, INFINITE);
        EnterCriticalSection(&pool->lock);
        if (pool->shutdown && !pool->head) { LeaveCriticalSection(&pool->lock); break; }
        VG_Task *t = pool->head;
        if (t) { pool->head = t->next; if (!pool->head) pool->tail = NULL; }
        LeaveCriticalSection(&pool->lock);
        if (t) {
            t->func(t->arg);
            free(t);
            EnterCriticalSection(&pool->lock);
            if (pool->n_pending && --pool->n_pending == 0) WakeAllConditionVariable(&pool->done_cond);
            LeaveCriticalSection(&pool->lock);
        }
    }
    return 0;
}
#else
static void *worker_thread(void *param) {
    VG_ThreadPool *pool = (VG_ThreadPool *)param;
    for (;;) {
        pthread_mutex_lock(&pool->lock);
        while (!pool->head && !pool->shutdown) pthread_cond_wait(&pool->cond, &pool->lock);
        if (pool->shutdown && !pool->head) { pthread_mutex_unlock(&pool->lock); break; }
        VG_Task *t = pool->head;
        if (t) { pool->head = t->next; if (!pool->head) pool->tail = NULL; }
        pthread_mutex_unlock(&pool->lock);
        if (t) {
            t->func(t->arg);
            free(t);
            pthread_mutex_lock(&pool->lock);
            if (pool->n_pending && --pool->n_pending == 0) pthread_cond_broadcast(&pool->done_cond);
            pthread_mutex_unlock(&pool->lock);
        }
    }
    return NULL;
}
#endif

VG_ThreadPool *vg_threadpool_create(unsigned n_workers) {
    if (n_workers == 0) n_workers = 1;
    VG_ThreadPool *pool = (VG_ThreadPool *)calloc(1, sizeof(*pool));
    if (!pool) return NULL;
    pool->n_workers = n_workers;
#ifdef _WIN32
    InitializeCriticalSection(&pool->lock);
    InitializeConditionVariable(&pool->done_cond);
    pool->semaphore = CreateSemaphoreW(NULL, 0, (LONG)n_workers, NULL);
    if (!pool->semaphore) { DeleteCriticalSection(&pool->lock); free(pool); return NULL; }
    pool->threads = (HANDLE *)calloc(n_workers, sizeof(HANDLE));
    if (!pool->threads) { CloseHandle(pool->semaphore); DeleteCriticalSection(&pool->lock); free(pool); return NULL; }
    for (unsigned i = 0; i < n_workers; ++i) pool->threads[i] = CreateThread(NULL, 0, worker_thread, pool, 0, NULL);
#else
    pthread_mutex_init(&pool->lock, NULL);
    pthread_cond_init(&pool->cond, NULL);
    pthread_cond_init(&pool->done_cond, NULL);
    pool->threads = (pthread_t *)calloc(n_workers, sizeof(pthread_t));
    if (!pool->threads) { pthread_mutex_destroy(&pool->lock); pthread_cond_destroy(&pool->cond); pthread_cond_destroy(&pool->done_cond); free(pool); return NULL; }
    for (unsigned i = 0; i < n_workers; ++i) pthread_create(&pool->threads[i], NULL, worker_thread, pool);
#endif
    return pool;
}

void vg_threadpool_destroy(VG_ThreadPool *pool) {
    if (!pool) return;
    vg_threadpool_wait(pool);
#ifdef _WIN32
    EnterCriticalSection(&pool->lock);
    pool->shutdown = 1;
    LeaveCriticalSection(&pool->lock);
    ReleaseSemaphore(pool->semaphore, (LONG)pool->n_workers, NULL);
    WaitForMultipleObjects(pool->n_workers, pool->threads, TRUE, INFINITE);
    for (unsigned i = 0; i < pool->n_workers; ++i) CloseHandle(pool->threads[i]);
    CloseHandle(pool->semaphore);
    DeleteCriticalSection(&pool->lock);
#else
    pthread_mutex_lock(&pool->lock);
    pool->shutdown = 1;
    pthread_cond_broadcast(&pool->cond);
    pthread_mutex_unlock(&pool->lock);
    for (unsigned i = 0; i < pool->n_workers; ++i) pthread_join(pool->threads[i], NULL);
    pthread_mutex_destroy(&pool->lock);
    pthread_cond_destroy(&pool->cond);
    pthread_cond_destroy(&pool->done_cond);
#endif
    free(pool->threads);
    VG_Task *t = pool->head;
    while (t) { VG_Task *next = t->next; free(t); t = next; }
    free(pool);
}

void vg_threadpool_submit(VG_ThreadPool *pool, VG_ThreadPoolFunc func, void *arg) {
    if (!pool || !func) return;
    VG_Task *t = (VG_Task *)calloc(1, sizeof(*t));
    if (!t) return;
    t->func = func; t->arg = arg;
#ifdef _WIN32
    EnterCriticalSection(&pool->lock);
    if (pool->tail) { pool->tail->next = t; pool->tail = t; }
    else { pool->head = pool->tail = t; }
    ++pool->n_pending;
    LeaveCriticalSection(&pool->lock);
    ReleaseSemaphore(pool->semaphore, 1, NULL);
#else
    pthread_mutex_lock(&pool->lock);
    if (pool->tail) { pool->tail->next = t; pool->tail = t; }
    else { pool->head = pool->tail = t; }
    ++pool->n_pending;
    pthread_cond_signal(&pool->cond);
    pthread_mutex_unlock(&pool->lock);
#endif
}

void vg_threadpool_wait(VG_ThreadPool *pool) {
    if (!pool) return;
#ifdef _WIN32
    EnterCriticalSection(&pool->lock);
    while (pool->n_pending > 0) SleepConditionVariableCS(&pool->done_cond, &pool->lock, INFINITE);
    LeaveCriticalSection(&pool->lock);
#else
    pthread_mutex_lock(&pool->lock);
    while (pool->n_pending > 0) pthread_cond_wait(&pool->done_cond, &pool->lock);
    pthread_mutex_unlock(&pool->lock);
#endif
}

unsigned vg_threadpool_workers(const VG_ThreadPool *pool) { return pool ? pool->n_workers : 0; }

typedef struct VG_ParallelJob {
    void (*fn)(void *arg, uint32_t begin, uint32_t end);
    void *arg;
    uint32_t begin;
    uint32_t end;
} VG_ParallelJob;

static void parallel_job_run(void *a) {
    VG_ParallelJob *job = (VG_ParallelJob *)a;
    job->fn(job->arg, job->begin, job->end);
    free(job);
}

void vg_threadpool_parallel_for(VG_ThreadPool *pool,
                                void (*fn)(void *arg, uint32_t begin, uint32_t end),
                                void *arg, uint32_t n_items) {
    if (!fn || n_items == 0) return;
    unsigned workers = pool ? pool->n_workers : 1;
    if (!pool || workers <= 1) { fn(arg, 0, n_items); return; }
    uint32_t chunks = workers < n_items ? workers : n_items;
    uint32_t per = (n_items + chunks - 1u) / chunks;
    for (uint32_t c = 1; c < chunks; ++c) {
        uint32_t begin = c * per;
        if (begin >= n_items) break;
        uint32_t end = begin + per > n_items ? n_items : begin + per;
        VG_ParallelJob *job = (VG_ParallelJob *)malloc(sizeof(*job));
        if (!job) continue;
        job->fn = fn; job->arg = arg; job->begin = begin; job->end = end;
        vg_threadpool_submit(pool, parallel_job_run, job);
    }
    fn(arg, 0, per > n_items ? n_items : per);
    vg_threadpool_wait(pool);
}
