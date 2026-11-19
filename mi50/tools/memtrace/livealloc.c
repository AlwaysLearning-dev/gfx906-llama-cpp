// livealloc: LD_PRELOAD shim that tracks live heap blocks >= LIVEALLOC_MIN bytes with their call stacks.
// kill -USR1 <pid> writes the live bytes per call stack to $LIVEALLOC_OUT.<n> (default /tmp/livealloc.<n>).
// Build: gcc -O2 -fPIC -shared -o livealloc.so livealloc.c -ldl -lpthread
#define _GNU_SOURCE
#include <dlfcn.h>
#include <execinfo.h>
#include <pthread.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <malloc.h>

#define DEPTH 14
#define NSLOT (1u << 25)
#define NSTACK (1u << 18)

static void *(*r_malloc)(size_t); static void (*r_free)(void *); static void *(*r_calloc)(size_t, size_t);
static void *(*r_realloc)(void *, size_t); static int (*r_posix_memalign)(void **, size_t, size_t);
static void *(*r_aligned_alloc)(size_t, size_t); static void *(*r_memalign)(size_t, size_t);

typedef struct { void *p; size_t size; uint32_t stack; } slot_t;
typedef struct { uint64_t h; int n; void *f[DEPTH]; size_t live; size_t count; size_t total; } cstack_t;
static slot_t *slots; static cstack_t *stacks;
static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
static __thread int busy;
static size_t min_size = 4096; static volatile int want_dump; static int n_dump; static int ready;
static char boot[1 << 16]; static size_t boot_used;

static void on_usr1(int s) { (void) s; want_dump = 1; }

static void init(void) {
    static int doing; if (ready || doing) return; doing = 1;
    r_malloc = dlsym(RTLD_NEXT, "malloc"); r_free = dlsym(RTLD_NEXT, "free"); r_calloc = dlsym(RTLD_NEXT, "calloc");
    r_realloc = dlsym(RTLD_NEXT, "realloc"); r_posix_memalign = dlsym(RTLD_NEXT, "posix_memalign");
    r_aligned_alloc = dlsym(RTLD_NEXT, "aligned_alloc"); r_memalign = dlsym(RTLD_NEXT, "memalign");
    const char *e = getenv("LIVEALLOC_MIN"); if (e) min_size = strtoul(e, 0, 10);
    busy = 1;
    slots = r_calloc(NSLOT, sizeof(slot_t)); stacks = r_calloc(NSTACK, sizeof(cstack_t));
    void *tmp[4]; backtrace(tmp, 4);   // loads libgcc now, not inside a tracked call
    busy = 0;
    signal(SIGUSR1, on_usr1);
    ready = 1;
}

static void dump(void) {
    char path[512]; const char *o = getenv("LIVEALLOC_OUT");
    snprintf(path, sizeof path, "%s.%d", o ? o : "/tmp/livealloc", ++n_dump);
    FILE *f = fopen(path, "w"); if (!f) return;
    size_t tot = 0; for (uint32_t i = 0; i < NSTACK; i++) tot += stacks[i].live;
    fprintf(f, "live tracked bytes: %zu\n", tot);
    struct mallinfo2 mi = mallinfo2();
    fprintf(f, "MALLINFO main_arena=%zu mmapped=%zu in_use=%zu free_in_heap=%zu top_releasable=%zu\n", mi.arena, mi.hblkhd, mi.uordblks, mi.fordblks, mi.keepcost);
    if (getenv("LIVEALLOC_TRIM")) { int r = malloc_trim(0); struct mallinfo2 m2 = mallinfo2(); fprintf(f, "TRIM returned=%d main_arena=%zu free_in_heap=%zu\n", r, m2.arena, m2.fordblks); }
    for (uint32_t i = 0; i < NSTACK; i++) {
        cstack_t *s = &stacks[i]; if (!s->n || (s->live < (1u << 20) && s->total < (64u << 20))) continue;
        fprintf(f, "STACK live=%zu blocks=%zu allocated_total=%zu\n", s->live, s->count, s->total);
        for (int k = 0; k < s->n; k++) {
            Dl_info d; if (dladdr(s->f[k], &d) && d.dli_fname)
                fprintf(f, "  %s+0x%lx %s\n", d.dli_fname, (unsigned long) ((char *) s->f[k] - (char *) d.dli_fbase), d.dli_sname ? d.dli_sname : "");
            else fprintf(f, "  ?? %p\n", s->f[k]);
        }
    }
    fclose(f);
}

static void track(void *p, size_t size) {
    if (!p || size < min_size || !ready || busy) return;
    busy = 1;
    void *fr[DEPTH + 2]; int n = backtrace(fr, DEPTH + 2); int skip = n > 2 ? 2 : 0; n -= skip;
    uint64_t h = 1469598103934665603ull; for (int k = 0; k < n; k++) { h ^= (uint64_t) fr[k + skip]; h *= 1099511628211ull; }
    pthread_mutex_lock(&mu);
    uint32_t si = (uint32_t) (h % NSTACK);
    for (uint32_t t = 0; t < NSTACK; t++, si = (si + 1) % NSTACK) {
        if (!stacks[si].n) { stacks[si].h = h; stacks[si].n = n; memcpy(stacks[si].f, fr + skip, n * sizeof(void *)); break; }
        if (stacks[si].h == h) break;
    }
    stacks[si].live += size; stacks[si].count++; stacks[si].total += size;
    uint32_t i = (uint32_t) (((uintptr_t) p >> 4) * 2654435761u) % NSLOT;
    for (uint32_t t = 0; t < NSLOT; t++, i = (i + 1) % NSLOT) if (!slots[i].p || slots[i].p == (void *) -1) { slots[i].p = p; slots[i].size = size; slots[i].stack = si; break; }
    if (want_dump) { want_dump = 0; dump(); }
    pthread_mutex_unlock(&mu);
    busy = 0;
}

static void untrack(void *p) {
    if (!p || !ready || busy) return;
    busy = 1;
    pthread_mutex_lock(&mu);
    uint32_t i = (uint32_t) (((uintptr_t) p >> 4) * 2654435761u) % NSLOT;
    for (uint32_t t = 0; t < NSLOT; t++, i = (i + 1) % NSLOT) {
        if (!slots[i].p) break;
        if (slots[i].p == p) { stacks[slots[i].stack].live -= slots[i].size; stacks[slots[i].stack].count--; slots[i].p = (void *) -1; break; }
    }
    pthread_mutex_unlock(&mu);
    busy = 0;
}

static int in_boot(void *p) { return (char *) p >= boot && (char *) p < boot + sizeof boot; }

void *malloc(size_t n) { if (!r_malloc) { init(); if (!r_malloc) { void *p = boot + boot_used; boot_used += (n + 15) & ~15ul; return p; } } void *p = r_malloc(n); track(p, n); return p; }
void *calloc(size_t a, size_t b) { if (!r_calloc) { void *p = boot + boot_used; boot_used += (a * b + 15) & ~15ul; return p; } void *p = r_calloc(a, b); track(p, a * b); return p; }
void free(void *p) { if (in_boot(p)) return; if (!r_free) init(); untrack(p); r_free(p); }
void *realloc(void *p, size_t n) { if (!r_realloc) init(); if (in_boot(p)) { void *q = r_malloc(n); memcpy(q, p, n); return q; } untrack(p); void *q = r_realloc(p, n); track(q, n); return q; }
int posix_memalign(void **o, size_t a, size_t n) { if (!r_posix_memalign) init(); int r = r_posix_memalign(o, a, n); if (!r) track(*o, n); return r; }
void *aligned_alloc(size_t a, size_t n) { if (!r_aligned_alloc) init(); void *p = r_aligned_alloc(a, n); track(p, n); return p; }
void *memalign(size_t a, size_t n) { if (!r_memalign) init(); void *p = r_memalign(a, n); track(p, n); return p; }
