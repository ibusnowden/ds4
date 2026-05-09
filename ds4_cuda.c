#include "ds4_cuda.h"

#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

typedef int CUdevice;
typedef void *CUcontext;
typedef uint64_t CUdeviceptr;
typedef void *CUstream;
typedef void *CUmodule;
typedef void *CUfunction;
typedef void *nvrtcProgram;

typedef int (*cuInit_fn)(unsigned int flags);
typedef int (*cuDeviceGetCount_fn)(int *count);
typedef int (*cuDeviceGet_fn)(CUdevice *device, int ordinal);
typedef int (*cuDeviceGetName_fn)(char *name, int len, CUdevice dev);
typedef int (*cuDeviceTotalMem_fn)(size_t *bytes, CUdevice dev);
typedef int (*cuCtxCreate_fn)(CUcontext *pctx, unsigned int flags, CUdevice dev);
typedef int (*cuCtxDestroy_fn)(CUcontext ctx);
typedef int (*cuMemGetInfo_fn)(size_t *free_bytes, size_t *total_bytes);
typedef int (*cuMemAlloc_fn)(CUdeviceptr *dptr, size_t bytesize);
typedef int (*cuMemFree_fn)(CUdeviceptr dptr);
typedef int (*cuMemHostRegister_fn)(void *p, size_t bytesize, unsigned int Flags);
typedef int (*cuMemHostUnregister_fn)(void *p);
typedef int (*cuMemHostGetDevicePointer_fn)(CUdeviceptr *pdptr, void *p, unsigned int Flags);
typedef int (*cuMemcpyHtoD_fn)(CUdeviceptr dstDevice, const void *srcHost, size_t byteCount);
typedef int (*cuMemcpyDtoH_fn)(void *dstHost, CUdeviceptr srcDevice, size_t byteCount);
typedef int (*cuMemcpyDtoD_fn)(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t byteCount);
typedef int (*cuStreamCreate_fn)(CUstream *phStream, unsigned int flags);
typedef int (*cuStreamDestroy_fn)(CUstream hStream);
typedef int (*cuStreamSynchronize_fn)(CUstream hStream);
typedef int (*cuModuleLoadData_fn)(CUmodule *module, const void *image);
typedef int (*cuModuleUnload_fn)(CUmodule hmod);
typedef int (*cuModuleGetFunction_fn)(CUfunction *hfunc, CUmodule hmod, const char *name);
typedef int (*cuLaunchKernel_fn)(CUfunction f,
                                 unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
                                 unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
                                 unsigned int sharedMemBytes, CUstream hStream,
                                 void **kernelParams, void **extra);
typedef int (*cuGetErrorString_fn)(int error, const char **pstr);
typedef int (*nvrtcCreateProgram_fn)(nvrtcProgram *prog, const char *src, const char *name,
                                     int numHeaders, const char * const *headers,
                                     const char * const *includeNames);
typedef int (*nvrtcDestroyProgram_fn)(nvrtcProgram *prog);
typedef int (*nvrtcCompileProgram_fn)(nvrtcProgram prog, int numOptions, const char * const *options);
typedef int (*nvrtcGetPTXSize_fn)(nvrtcProgram prog, size_t *ptxSizeRet);
typedef int (*nvrtcGetPTX_fn)(nvrtcProgram prog, char *ptx);
typedef int (*nvrtcGetCUBINSize_fn)(nvrtcProgram prog, size_t *cubinSizeRet);
typedef int (*nvrtcGetCUBIN_fn)(nvrtcProgram prog, char *cubin);
typedef int (*nvrtcGetProgramLogSize_fn)(nvrtcProgram prog, size_t *logSizeRet);
typedef int (*nvrtcGetProgramLog_fn)(nvrtcProgram prog, char *log);
typedef const char *(*nvrtcGetErrorString_fn)(int result);

struct ds4_cuda_tensor {
    CUdeviceptr ptr;
    uint64_t bytes;
    bool owner;
};

struct ds4_cuda_module {
    CUmodule module;
};

struct ds4_cuda_kernel {
    CUfunction function;
};

struct ds4_cuda_host_map {
    void *registered_host;
    uint64_t registered_bytes;
    uint64_t delta;
    CUdeviceptr device_ptr;
    uint64_t bytes;
};

static void *g_cuda_lib;
static CUcontext g_cuda_ctx;
static CUstream g_cuda_stream;
static ds4_cuda_info g_info;
static bool g_ready;

static cuInit_fn p_cuInit;
static cuDeviceGetCount_fn p_cuDeviceGetCount;
static cuDeviceGet_fn p_cuDeviceGet;
static cuDeviceGetName_fn p_cuDeviceGetName;
static cuDeviceTotalMem_fn p_cuDeviceTotalMem;
static cuCtxCreate_fn p_cuCtxCreate;
static cuCtxDestroy_fn p_cuCtxDestroy;
static cuMemGetInfo_fn p_cuMemGetInfo;
static cuMemAlloc_fn p_cuMemAlloc;
static cuMemFree_fn p_cuMemFree;
static cuMemHostRegister_fn p_cuMemHostRegister;
static cuMemHostUnregister_fn p_cuMemHostUnregister;
static cuMemHostGetDevicePointer_fn p_cuMemHostGetDevicePointer;
static cuMemcpyHtoD_fn p_cuMemcpyHtoD;
static cuMemcpyDtoH_fn p_cuMemcpyDtoH;
static cuMemcpyDtoD_fn p_cuMemcpyDtoD;
static cuStreamCreate_fn p_cuStreamCreate;
static cuStreamDestroy_fn p_cuStreamDestroy;
static cuStreamSynchronize_fn p_cuStreamSynchronize;
static cuModuleLoadData_fn p_cuModuleLoadData;
static cuModuleUnload_fn p_cuModuleUnload;
static cuModuleGetFunction_fn p_cuModuleGetFunction;
static cuLaunchKernel_fn p_cuLaunchKernel;
static cuGetErrorString_fn p_cuGetErrorString;

static void *g_nvrtc_lib;
static nvrtcCreateProgram_fn p_nvrtcCreateProgram;
static nvrtcDestroyProgram_fn p_nvrtcDestroyProgram;
static nvrtcCompileProgram_fn p_nvrtcCompileProgram;
static nvrtcGetPTXSize_fn p_nvrtcGetPTXSize;
static nvrtcGetPTX_fn p_nvrtcGetPTX;
static nvrtcGetCUBINSize_fn p_nvrtcGetCUBINSize;
static nvrtcGetCUBIN_fn p_nvrtcGetCUBIN;
static nvrtcGetProgramLogSize_fn p_nvrtcGetProgramLogSize;
static nvrtcGetProgramLog_fn p_nvrtcGetProgramLog;
static nvrtcGetErrorString_fn p_nvrtcGetErrorString;

static void set_err(char *err, size_t errlen, const char *msg) {
    if (errlen != 0) snprintf(err, errlen, "%s", msg);
}

static void set_cuda_err(char *err, size_t errlen, const char *where, int rc) {
    const char *detail = NULL;
    if (p_cuGetErrorString) (void)p_cuGetErrorString(rc, &detail);
    if (!detail) detail = "unknown CUDA driver error";
    if (errlen != 0) snprintf(err, errlen, "%s failed: %s (%d)", where, detail, rc);
}

static int load_symbol(void **out, const char *name, char *err, size_t errlen) {
    *out = dlsym(g_cuda_lib, name);
    if (!*out) {
        if (errlen != 0) snprintf(err, errlen, "CUDA driver symbol not found: %s", name);
        return 0;
    }
    return 1;
}

static int load_nvrtc_symbol(void **out, const char *name, char *err, size_t errlen) {
    *out = dlsym(g_nvrtc_lib, name);
    if (!*out) {
        if (errlen != 0) snprintf(err, errlen, "NVRTC symbol not found: %s", name);
        return 0;
    }
    return 1;
}

static void unload_nvrtc(void) {
    if (g_nvrtc_lib) dlclose(g_nvrtc_lib);
    g_nvrtc_lib = NULL;
    p_nvrtcCreateProgram = NULL;
    p_nvrtcDestroyProgram = NULL;
    p_nvrtcCompileProgram = NULL;
    p_nvrtcGetPTXSize = NULL;
    p_nvrtcGetPTX = NULL;
    p_nvrtcGetCUBINSize = NULL;
    p_nvrtcGetCUBIN = NULL;
    p_nvrtcGetProgramLogSize = NULL;
    p_nvrtcGetProgramLog = NULL;
    p_nvrtcGetErrorString = NULL;
}

static int ensure_nvrtc(char *err, size_t errlen) {
    if (g_nvrtc_lib) return 1;
    static const char *names[] = {
        "libnvrtc.so",
        "libnvrtc.so.12",
        "libnvrtc.so.11",
        "libnvrtc.so.10.2",
        NULL,
    };
    for (int i = 0; names[i]; i++) {
        g_nvrtc_lib = dlopen(names[i], RTLD_NOW | RTLD_LOCAL);
        if (g_nvrtc_lib) break;
    }
    if (!g_nvrtc_lib) {
        set_err(err, errlen,
                "libnvrtc is not available; CUDA kernel source compilation requires the CUDA toolkit runtime");
        return 0;
    }
    if (!load_nvrtc_symbol((void **)&p_nvrtcCreateProgram, "nvrtcCreateProgram", err, errlen) ||
        !load_nvrtc_symbol((void **)&p_nvrtcDestroyProgram, "nvrtcDestroyProgram", err, errlen) ||
        !load_nvrtc_symbol((void **)&p_nvrtcCompileProgram, "nvrtcCompileProgram", err, errlen) ||
        !load_nvrtc_symbol((void **)&p_nvrtcGetPTXSize, "nvrtcGetPTXSize", err, errlen) ||
        !load_nvrtc_symbol((void **)&p_nvrtcGetPTX, "nvrtcGetPTX", err, errlen) ||
        !load_nvrtc_symbol((void **)&p_nvrtcGetProgramLogSize, "nvrtcGetProgramLogSize", err, errlen) ||
        !load_nvrtc_symbol((void **)&p_nvrtcGetProgramLog, "nvrtcGetProgramLog", err, errlen) ||
        !load_nvrtc_symbol((void **)&p_nvrtcGetErrorString, "nvrtcGetErrorString", err, errlen))
    {
        unload_nvrtc();
        return 0;
    }
    p_nvrtcGetCUBINSize = (nvrtcGetCUBINSize_fn)dlsym(g_nvrtc_lib, "nvrtcGetCUBINSize");
    p_nvrtcGetCUBIN = (nvrtcGetCUBIN_fn)dlsym(g_nvrtc_lib, "nvrtcGetCUBIN");
    return 1;
}

static void set_nvrtc_err(char *err, size_t errlen, const char *where, int rc) {
    const char *detail = p_nvrtcGetErrorString ? p_nvrtcGetErrorString(rc) : NULL;
    if (!detail) detail = "unknown NVRTC error";
    if (errlen != 0) snprintf(err, errlen, "%s failed: %s (%d)", where, detail, rc);
}

static bool tensor_bounds_ok(const ds4_cuda_tensor *tensor, uint64_t offset, uint64_t bytes) {
    return tensor && offset <= tensor->bytes && bytes <= tensor->bytes - offset;
}

static uint64_t host_page_size(void) {
    long p = sysconf(_SC_PAGESIZE);
    return p > 0 ? (uint64_t)p : 4096u;
}

static bool checked_add_u64(uint64_t a, uint64_t b, uint64_t *out) {
    if (a > UINT64_MAX - b) return false;
    *out = a + b;
    return true;
}

int ds4_cuda_init(ds4_cuda_info *info, char *err, size_t errlen) {
    if (g_ready) {
        if (info) *info = g_info;
        return 1;
    }

    g_cuda_lib = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
    if (!g_cuda_lib) {
        set_err(err, errlen, "libcuda.so.1 is not available; run inside a Slurm CUDA allocation");
        return 0;
    }

    if (!load_symbol((void **)&p_cuInit, "cuInit", err, errlen) ||
        !load_symbol((void **)&p_cuDeviceGetCount, "cuDeviceGetCount", err, errlen) ||
        !load_symbol((void **)&p_cuDeviceGet, "cuDeviceGet", err, errlen) ||
        !load_symbol((void **)&p_cuDeviceGetName, "cuDeviceGetName", err, errlen) ||
        !load_symbol((void **)&p_cuDeviceTotalMem, "cuDeviceTotalMem_v2", err, errlen) ||
        !load_symbol((void **)&p_cuCtxCreate, "cuCtxCreate_v2", err, errlen) ||
        !load_symbol((void **)&p_cuCtxDestroy, "cuCtxDestroy_v2", err, errlen) ||
        !load_symbol((void **)&p_cuMemGetInfo, "cuMemGetInfo_v2", err, errlen) ||
        !load_symbol((void **)&p_cuMemAlloc, "cuMemAlloc_v2", err, errlen) ||
        !load_symbol((void **)&p_cuMemFree, "cuMemFree_v2", err, errlen) ||
        !load_symbol((void **)&p_cuMemHostRegister, "cuMemHostRegister_v2", err, errlen) ||
        !load_symbol((void **)&p_cuMemHostUnregister, "cuMemHostUnregister", err, errlen) ||
        !load_symbol((void **)&p_cuMemHostGetDevicePointer, "cuMemHostGetDevicePointer_v2", err, errlen) ||
        !load_symbol((void **)&p_cuMemcpyHtoD, "cuMemcpyHtoD_v2", err, errlen) ||
        !load_symbol((void **)&p_cuMemcpyDtoH, "cuMemcpyDtoH_v2", err, errlen) ||
        !load_symbol((void **)&p_cuMemcpyDtoD, "cuMemcpyDtoD_v2", err, errlen) ||
        !load_symbol((void **)&p_cuStreamCreate, "cuStreamCreate", err, errlen) ||
        !load_symbol((void **)&p_cuStreamDestroy, "cuStreamDestroy_v2", err, errlen) ||
        !load_symbol((void **)&p_cuStreamSynchronize, "cuStreamSynchronize", err, errlen) ||
        !load_symbol((void **)&p_cuModuleLoadData, "cuModuleLoadData", err, errlen) ||
        !load_symbol((void **)&p_cuModuleUnload, "cuModuleUnload", err, errlen) ||
        !load_symbol((void **)&p_cuModuleGetFunction, "cuModuleGetFunction", err, errlen) ||
        !load_symbol((void **)&p_cuLaunchKernel, "cuLaunchKernel", err, errlen) ||
        !load_symbol((void **)&p_cuGetErrorString, "cuGetErrorString", err, errlen))
    {
        ds4_cuda_cleanup();
        return 0;
    }

    int rc = p_cuInit(0);
    if (rc != 0) {
        set_cuda_err(err, errlen, "cuInit", rc);
        ds4_cuda_cleanup();
        return 0;
    }

    int count = 0;
    rc = p_cuDeviceGetCount(&count);
    if (rc != 0) {
        set_cuda_err(err, errlen, "cuDeviceGetCount", rc);
        ds4_cuda_cleanup();
        return 0;
    }
    if (count <= 0) {
        set_err(err, errlen, "CUDA driver loaded but no CUDA devices are visible");
        ds4_cuda_cleanup();
        return 0;
    }

    CUdevice dev = 0;
    rc = p_cuDeviceGet(&dev, 0);
    if (rc != 0) {
        set_cuda_err(err, errlen, "cuDeviceGet", rc);
        ds4_cuda_cleanup();
        return 0;
    }

    memset(&g_info, 0, sizeof(g_info));
    g_info.device_count = count;
    g_info.selected_device = 0;
    rc = p_cuDeviceGetName(g_info.name, (int)sizeof(g_info.name), dev);
    if (rc != 0) {
        set_cuda_err(err, errlen, "cuDeviceGetName", rc);
        ds4_cuda_cleanup();
        return 0;
    }

    size_t total = 0;
    rc = p_cuDeviceTotalMem(&total, dev);
    if (rc != 0) {
        set_cuda_err(err, errlen, "cuDeviceTotalMem", rc);
        ds4_cuda_cleanup();
        return 0;
    }
    g_info.total_mem = (uint64_t)total;

    rc = p_cuCtxCreate(&g_cuda_ctx, 0x08u, dev);
    if (rc != 0) {
        set_cuda_err(err, errlen, "cuCtxCreate", rc);
        ds4_cuda_cleanup();
        return 0;
    }

    size_t free_bytes = 0;
    size_t total_ctx = 0;
    rc = p_cuMemGetInfo(&free_bytes, &total_ctx);
    if (rc == 0) {
        g_info.free_mem = (uint64_t)free_bytes;
        if (total_ctx != 0) g_info.total_mem = (uint64_t)total_ctx;
    }

    rc = p_cuStreamCreate(&g_cuda_stream, 0);
    if (rc != 0) {
        set_cuda_err(err, errlen, "cuStreamCreate", rc);
        ds4_cuda_cleanup();
        return 0;
    }

    g_ready = true;
    if (info) *info = g_info;
    return 1;
}

void ds4_cuda_cleanup(void) {
    if (g_cuda_stream && p_cuStreamDestroy) {
        (void)p_cuStreamDestroy(g_cuda_stream);
    }
    g_cuda_stream = NULL;
    if (g_cuda_ctx && p_cuCtxDestroy) {
        (void)p_cuCtxDestroy(g_cuda_ctx);
    }
    g_cuda_ctx = NULL;
    g_ready = false;
    memset(&g_info, 0, sizeof(g_info));
    if (g_cuda_lib) dlclose(g_cuda_lib);
    g_cuda_lib = NULL;
    unload_nvrtc();
    p_cuInit = NULL;
    p_cuDeviceGetCount = NULL;
    p_cuDeviceGet = NULL;
    p_cuDeviceGetName = NULL;
    p_cuDeviceTotalMem = NULL;
    p_cuCtxCreate = NULL;
    p_cuCtxDestroy = NULL;
    p_cuMemGetInfo = NULL;
    p_cuMemAlloc = NULL;
    p_cuMemFree = NULL;
    p_cuMemHostRegister = NULL;
    p_cuMemHostUnregister = NULL;
    p_cuMemHostGetDevicePointer = NULL;
    p_cuMemcpyHtoD = NULL;
    p_cuMemcpyDtoH = NULL;
    p_cuMemcpyDtoD = NULL;
    p_cuStreamCreate = NULL;
    p_cuStreamDestroy = NULL;
    p_cuStreamSynchronize = NULL;
    p_cuModuleLoadData = NULL;
    p_cuModuleUnload = NULL;
    p_cuModuleGetFunction = NULL;
    p_cuLaunchKernel = NULL;
    p_cuGetErrorString = NULL;
}

bool ds4_cuda_ready(void) {
    return g_ready;
}

const ds4_cuda_info *ds4_cuda_get_info(void) {
    return g_ready ? &g_info : NULL;
}

int ds4_cuda_synchronize(char *err, size_t errlen) {
    if (!g_ready || !g_cuda_stream) {
        set_err(err, errlen, "CUDA runtime is not initialized");
        return 0;
    }
    int rc = p_cuStreamSynchronize(g_cuda_stream);
    if (rc != 0) {
        set_cuda_err(err, errlen, "cuStreamSynchronize", rc);
        return 0;
    }
    return 1;
}

ds4_cuda_tensor *ds4_cuda_tensor_alloc(uint64_t bytes) {
    if (!g_ready) return NULL;
    ds4_cuda_tensor *tensor = calloc(1, sizeof(*tensor));
    if (!tensor) return NULL;
    size_t alloc_bytes = bytes ? (size_t)bytes : 1u;
    if ((uint64_t)alloc_bytes != (bytes ? bytes : 1u)) {
        free(tensor);
        return NULL;
    }
    CUdeviceptr ptr = 0;
    int rc = p_cuMemAlloc(&ptr, alloc_bytes);
    if (rc != 0) {
        free(tensor);
        return NULL;
    }
    tensor->ptr = ptr;
    tensor->bytes = bytes;
    tensor->owner = true;
    return tensor;
}

ds4_cuda_tensor *ds4_cuda_tensor_view(const ds4_cuda_tensor *base, uint64_t offset, uint64_t bytes) {
    if (!tensor_bounds_ok(base, offset, bytes)) return NULL;
    ds4_cuda_tensor *tensor = calloc(1, sizeof(*tensor));
    if (!tensor) return NULL;
    tensor->ptr = base->ptr + offset;
    tensor->bytes = bytes;
    tensor->owner = false;
    return tensor;
}

void ds4_cuda_tensor_free(ds4_cuda_tensor *tensor) {
    if (!tensor) return;
    if (tensor->owner && tensor->ptr && p_cuMemFree) {
        (void)p_cuMemFree(tensor->ptr);
    }
    free(tensor);
}

uint64_t ds4_cuda_tensor_bytes(const ds4_cuda_tensor *tensor) {
    return tensor ? tensor->bytes : 0;
}

uint64_t ds4_cuda_tensor_device_ptr(const ds4_cuda_tensor *tensor) {
    return tensor ? tensor->ptr : 0;
}

int ds4_cuda_tensor_write(ds4_cuda_tensor *tensor, uint64_t offset, const void *data, uint64_t bytes) {
    if (!tensor_bounds_ok(tensor, offset, bytes) || (bytes != 0 && !data) || !p_cuMemcpyHtoD) return 0;
    if (bytes == 0) return 1;
    return p_cuMemcpyHtoD(tensor->ptr + offset, data, (size_t)bytes) == 0;
}

int ds4_cuda_tensor_read(const ds4_cuda_tensor *tensor, uint64_t offset, void *data, uint64_t bytes) {
    if (!tensor_bounds_ok(tensor, offset, bytes) || (bytes != 0 && !data) || !p_cuMemcpyDtoH) return 0;
    if (bytes == 0) return 1;
    return p_cuMemcpyDtoH(data, tensor->ptr + offset, (size_t)bytes) == 0;
}

int ds4_cuda_tensor_copy(ds4_cuda_tensor *dst, uint64_t dst_offset,
                         const ds4_cuda_tensor *src, uint64_t src_offset,
                         uint64_t bytes) {
    if (!tensor_bounds_ok(dst, dst_offset, bytes) ||
        !tensor_bounds_ok(src, src_offset, bytes) ||
        !p_cuMemcpyDtoD)
    {
        return 0;
    }
    if (bytes == 0) return 1;
    return p_cuMemcpyDtoD(dst->ptr + dst_offset, src->ptr + src_offset, (size_t)bytes) == 0;
}

int ds4_cuda_host_register(ds4_cuda_host_map **out, const void *host, uint64_t bytes,
                           char *err, size_t errlen) {
    if (!out) return 0;
    *out = NULL;
    if (!g_ready || !p_cuMemHostRegister || !p_cuMemHostGetDevicePointer) {
        set_err(err, errlen, "CUDA runtime is not initialized for host-mapped memory");
        return 0;
    }
    if (!host || bytes == 0) {
        set_err(err, errlen, "invalid CUDA host map range");
        return 0;
    }

    const uint64_t page = host_page_size();
    const uint64_t h = (uint64_t)(uintptr_t)host;
    const uint64_t start = h & ~(page - 1u);
    const uint64_t delta = h - start;
    uint64_t end = 0;
    if (!checked_add_u64(h, bytes, &end)) {
        set_err(err, errlen, "CUDA host map range overflows address space");
        return 0;
    }
    const uint64_t end_aligned = (end + page - 1u) & ~(page - 1u);
    if (end_aligned < start) {
        set_err(err, errlen, "CUDA host map alignment overflow");
        return 0;
    }
    const uint64_t registered_bytes = end_aligned - start;
    if ((size_t)registered_bytes != registered_bytes) {
        set_err(err, errlen, "CUDA host map range is too large for this platform");
        return 0;
    }

    ds4_cuda_host_map *map = calloc(1, sizeof(*map));
    if (!map) {
        set_err(err, errlen, "out of memory allocating CUDA host map");
        return 0;
    }

    void *registered_host = (void *)(uintptr_t)start;
    int rc = p_cuMemHostRegister(registered_host, (size_t)registered_bytes, 0x02u | 0x08u);
    if (rc != 0) {
        set_cuda_err(err, errlen, "cuMemHostRegister", rc);
        free(map);
        return 0;
    }

    CUdeviceptr device = 0;
    rc = p_cuMemHostGetDevicePointer(&device, registered_host, 0);
    if (rc != 0) {
        set_cuda_err(err, errlen, "cuMemHostGetDevicePointer", rc);
        (void)p_cuMemHostUnregister(registered_host);
        free(map);
        return 0;
    }

    map->registered_host = registered_host;
    map->registered_bytes = registered_bytes;
    map->delta = delta;
    map->device_ptr = device + delta;
    map->bytes = bytes;
    *out = map;
    return 1;
}

void ds4_cuda_host_unregister(ds4_cuda_host_map *map) {
    if (!map) return;
    if (map->registered_host && p_cuMemHostUnregister) {
        (void)p_cuMemHostUnregister(map->registered_host);
    }
    free(map);
}

uint64_t ds4_cuda_host_device_ptr(const ds4_cuda_host_map *map) {
    return map ? map->device_ptr : 0;
}

uint64_t ds4_cuda_host_bytes(const ds4_cuda_host_map *map) {
    return map ? map->bytes : 0;
}

int ds4_cuda_module_load_data(ds4_cuda_module **out, const void *image, char *err, size_t errlen) {
    if (!out) return 0;
    *out = NULL;
    if (!g_ready || !p_cuModuleLoadData) {
        set_err(err, errlen, "CUDA runtime is not initialized");
        return 0;
    }
    if (!image) {
        set_err(err, errlen, "CUDA module image is null");
        return 0;
    }
    ds4_cuda_module *module = calloc(1, sizeof(*module));
    if (!module) {
        set_err(err, errlen, "out of memory allocating CUDA module wrapper");
        return 0;
    }
    int rc = p_cuModuleLoadData(&module->module, image);
    if (rc != 0) {
        set_cuda_err(err, errlen, "cuModuleLoadData", rc);
        free(module);
        return 0;
    }
    *out = module;
    return 1;
}

int ds4_cuda_module_load_source(ds4_cuda_module **out, const char *source, const char *name,
                                char *err, size_t errlen) {
    if (!out) return 0;
    *out = NULL;
    if (!source || !source[0]) {
        set_err(err, errlen, "CUDA source image is empty");
        return 0;
    }
    if (!ensure_nvrtc(err, errlen)) return 0;

    nvrtcProgram prog = NULL;
    int rc = p_nvrtcCreateProgram(&prog, source, name ? name : "ds4_cuda_kernels.cu", 0, NULL, NULL);
    if (rc != 0) {
        set_nvrtc_err(err, errlen, "nvrtcCreateProgram", rc);
        return 0;
    }

    const char *arch = getenv("DS4_CUDA_ARCH");
    if (!arch || !arch[0]) arch = "sm_89";
    char arch_opt[64];
    snprintf(arch_opt, sizeof(arch_opt), "--gpu-architecture=%s", arch);
    const char *opts[] = {
        "--std=c++11",
        "--use_fast_math",
        arch_opt,
    };
    rc = p_nvrtcCompileProgram(prog, (int)(sizeof(opts) / sizeof(opts[0])), opts);
    if (rc != 0) {
        size_t log_size = 0;
        (void)p_nvrtcGetProgramLogSize(prog, &log_size);
        char *log = NULL;
        if (log_size > 1) {
            log = malloc(log_size);
            if (log && p_nvrtcGetProgramLog(prog, log) == 0) {
                if (errlen != 0) snprintf(err, errlen, "nvrtcCompileProgram failed: %s", log);
            }
        }
        if (!log || errlen == 0) set_nvrtc_err(err, errlen, "nvrtcCompileProgram", rc);
        free(log);
        (void)p_nvrtcDestroyProgram(&prog);
        return 0;
    }

    if (p_nvrtcGetCUBINSize && p_nvrtcGetCUBIN && strncmp(arch, "sm_", 3) == 0) {
        size_t cubin_size = 0;
        rc = p_nvrtcGetCUBINSize(prog, &cubin_size);
        if (rc == 0 && cubin_size > 0) {
            char *cubin = malloc(cubin_size);
            if (!cubin) {
                set_err(err, errlen, "out of memory reading NVRTC CUBIN");
                (void)p_nvrtcDestroyProgram(&prog);
                return 0;
            }
            rc = p_nvrtcGetCUBIN(prog, cubin);
            (void)p_nvrtcDestroyProgram(&prog);
            if (rc != 0) {
                free(cubin);
                set_nvrtc_err(err, errlen, "nvrtcGetCUBIN", rc);
                return 0;
            }
            int ok = ds4_cuda_module_load_data(out, cubin, err, errlen);
            free(cubin);
            return ok;
        }
    }

    size_t ptx_size = 0;
    rc = p_nvrtcGetPTXSize(prog, &ptx_size);
    if (rc != 0) {
        set_nvrtc_err(err, errlen, "nvrtcGetPTXSize", rc);
        (void)p_nvrtcDestroyProgram(&prog);
        return 0;
    }
    if (ptx_size == 0) {
        set_err(err, errlen, "nvrtcGetPTXSize returned an empty PTX image");
        (void)p_nvrtcDestroyProgram(&prog);
        return 0;
    }
    char *ptx = malloc(ptx_size);
    if (!ptx) {
        set_err(err, errlen, "out of memory allocating CUDA PTX");
        (void)p_nvrtcDestroyProgram(&prog);
        return 0;
    }
    rc = p_nvrtcGetPTX(prog, ptx);
    (void)p_nvrtcDestroyProgram(&prog);
    if (rc != 0) {
        set_nvrtc_err(err, errlen, "nvrtcGetPTX", rc);
        free(ptx);
        return 0;
    }

    int ok = ds4_cuda_module_load_data(out, ptx, err, errlen);
    free(ptx);
    return ok;
}

void ds4_cuda_module_free(ds4_cuda_module *module) {
    if (!module) return;
    if (module->module && p_cuModuleUnload) (void)p_cuModuleUnload(module->module);
    free(module);
}

int ds4_cuda_module_get_kernel(ds4_cuda_kernel **out, ds4_cuda_module *module,
                               const char *name, char *err, size_t errlen) {
    if (!out) return 0;
    *out = NULL;
    if (!module || !module->module || !name || !name[0]) {
        set_err(err, errlen, "invalid CUDA module kernel lookup");
        return 0;
    }
    ds4_cuda_kernel *kernel = calloc(1, sizeof(*kernel));
    if (!kernel) {
        set_err(err, errlen, "out of memory allocating CUDA kernel wrapper");
        return 0;
    }
    int rc = p_cuModuleGetFunction(&kernel->function, module->module, name);
    if (rc != 0) {
        set_cuda_err(err, errlen, "cuModuleGetFunction", rc);
        free(kernel);
        return 0;
    }
    *out = kernel;
    return 1;
}

void ds4_cuda_kernel_free(ds4_cuda_kernel *kernel) {
    free(kernel);
}

int ds4_cuda_launch_kernel(ds4_cuda_kernel *kernel,
                           unsigned int grid_x, unsigned int grid_y, unsigned int grid_z,
                           unsigned int block_x, unsigned int block_y, unsigned int block_z,
                           unsigned int shared_mem_bytes,
                           void **args,
                           char *err, size_t errlen) {
    if (!g_ready || !kernel || !kernel->function) {
        set_err(err, errlen, "CUDA kernel launch requested before runtime/kernel initialization");
        return 0;
    }
    if (grid_x == 0 || grid_y == 0 || grid_z == 0 ||
        block_x == 0 || block_y == 0 || block_z == 0)
    {
        set_err(err, errlen, "CUDA kernel launch has zero grid or block dimension");
        return 0;
    }
    int rc = p_cuLaunchKernel(kernel->function,
                              grid_x, grid_y, grid_z,
                              block_x, block_y, block_z,
                              shared_mem_bytes,
                              g_cuda_stream,
                              args,
                              NULL);
    if (rc != 0) {
        set_cuda_err(err, errlen, "cuLaunchKernel", rc);
        return 0;
    }
    return 1;
}
