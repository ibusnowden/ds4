#include <stdio.h>
#include <dlfcn.h>
#include <stdlib.h>
#include <time.h>
typedef unsigned long long CUdeviceptr;
typedef void* CUcontext;
static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }
int main(void){
  void*h=dlopen("libcuda.so.1",RTLD_NOW); if(!h){printf("no libcuda\n");return 1;}
  int (*cuInit)(unsigned)=dlsym(h,"cuInit");
  int (*cuGet)(int*,int)=dlsym(h,"cuDeviceGet");
  int (*cuCtxCreate)(CUcontext*,unsigned,int)=dlsym(h,"cuCtxCreate_v2");
  int (*cuCtxSet)(CUcontext)=dlsym(h,"cuCtxSetCurrent");
  int (*cuEnablePeer)(CUcontext,unsigned)=dlsym(h,"cuCtxEnablePeerAccess");
  int (*cuMemAlloc)(CUdeviceptr*,size_t)=dlsym(h,"cuMemAlloc_v2");
  int (*cuMemcpyDtoD)(CUdeviceptr,CUdeviceptr,size_t)=dlsym(h,"cuMemcpyDtoD_v2");
  int (*cuMemcpyHtoD)(CUdeviceptr,const void*,size_t)=dlsym(h,"cuMemcpyHtoD_v2");
  int (*cuCtxSync)(void)=dlsym(h,"cuCtxSynchronize");
  cuInit(0);
  int d0,d1; cuGet(&d0,0); cuGet(&d1,1);
  CUcontext c0,c1; cuCtxCreate(&c0,0,d0); cuCtxCreate(&c1,0,d1);
  cuCtxSet(c0); int pe=cuEnablePeer(c1,0); printf("enablePeer ctx0->ctx1 rc=%d\n",pe);
  size_t sz=1ull<<30;
  CUdeviceptr p0,p1;
  cuCtxSet(c1); cuMemAlloc(&p1,sz);
  cuCtxSet(c0); cuMemAlloc(&p0,sz);
  void*hb=malloc(sz);
  int N=20;
  for(int i=0;i<3;i++) cuMemcpyDtoD(p0,p1,sz); cuCtxSync();
  double t=now(); for(int i=0;i<N;i++) cuMemcpyDtoD(p0,p1,sz); cuCtxSync();
  double dt=now()-t; printf("P2P DtoD dev1->dev0: %.1f GB/s\n",(double)N*sz/dt/1e9);
  for(int i=0;i<3;i++) cuMemcpyHtoD(p0,hb,sz); cuCtxSync();
  t=now(); for(int i=0;i<N;i++) cuMemcpyHtoD(p0,hb,sz); cuCtxSync();
  dt=now()-t; printf("HtoD host->dev0 (unpinned): %.1f GB/s\n",(double)N*sz/dt/1e9);
  return 0;
}
