#ifndef DS4_CUDA_H
#define DS4_CUDA_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

typedef struct {
    int device_count;
    int selected_device;
    char name[128];
    uint64_t total_mem;
    uint64_t free_mem;
} ds4_cuda_info;

typedef struct ds4_cuda_tensor ds4_cuda_tensor;
typedef struct ds4_cuda_module ds4_cuda_module;
typedef struct ds4_cuda_kernel ds4_cuda_kernel;
typedef struct ds4_cuda_host_map ds4_cuda_host_map;

int ds4_cuda_init(ds4_cuda_info *info, char *err, size_t errlen);
void ds4_cuda_cleanup(void);
bool ds4_cuda_ready(void);
const ds4_cuda_info *ds4_cuda_get_info(void);
int ds4_cuda_synchronize(char *err, size_t errlen);

ds4_cuda_tensor *ds4_cuda_tensor_alloc(uint64_t bytes);
ds4_cuda_tensor *ds4_cuda_tensor_view(const ds4_cuda_tensor *base, uint64_t offset, uint64_t bytes);
void ds4_cuda_tensor_free(ds4_cuda_tensor *tensor);
uint64_t ds4_cuda_tensor_bytes(const ds4_cuda_tensor *tensor);
uint64_t ds4_cuda_tensor_device_ptr(const ds4_cuda_tensor *tensor);
int ds4_cuda_tensor_write(ds4_cuda_tensor *tensor, uint64_t offset, const void *data, uint64_t bytes);
int ds4_cuda_tensor_read(const ds4_cuda_tensor *tensor, uint64_t offset, void *data, uint64_t bytes);
int ds4_cuda_tensor_copy(ds4_cuda_tensor *dst, uint64_t dst_offset,
                         const ds4_cuda_tensor *src, uint64_t src_offset,
                         uint64_t bytes);

int ds4_cuda_host_register(ds4_cuda_host_map **out, const void *host, uint64_t bytes,
                           char *err, size_t errlen);
void ds4_cuda_host_unregister(ds4_cuda_host_map *map);
uint64_t ds4_cuda_host_device_ptr(const ds4_cuda_host_map *map);
uint64_t ds4_cuda_host_bytes(const ds4_cuda_host_map *map);

int ds4_cuda_module_load_data(ds4_cuda_module **out, const void *image, char *err, size_t errlen);
int ds4_cuda_module_load_source(ds4_cuda_module **out, const char *source, const char *name,
                                char *err, size_t errlen);
void ds4_cuda_module_free(ds4_cuda_module *module);
int ds4_cuda_module_get_kernel(ds4_cuda_kernel **out, ds4_cuda_module *module,
                               const char *name, char *err, size_t errlen);
void ds4_cuda_kernel_free(ds4_cuda_kernel *kernel);
int ds4_cuda_launch_kernel(ds4_cuda_kernel *kernel,
                           unsigned int grid_x, unsigned int grid_y, unsigned int grid_z,
                           unsigned int block_x, unsigned int block_y, unsigned int block_z,
                           unsigned int shared_mem_bytes,
                           void **args,
                           char *err, size_t errlen);

#endif
