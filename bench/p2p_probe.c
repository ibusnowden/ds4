#include <stdio.h>
#include <dlfcn.h>
int main(void){
  void*h=dlopen("libcuda.so.1",RTLD_NOW);
  if(!h){printf("no libcuda: %s\n",dlerror());return 1;}
  int (*cuInit)(unsigned)=dlsym(h,"cuInit");
  int (*cuGet)(int*,int)=dlsym(h,"cuDeviceGet");
  int (*cuCount)(int*)=dlsym(h,"cuDeviceGetCount");
  int (*cuCanPeer)(int*,int,int)=dlsym(h,"cuDeviceCanAccessPeer");
  int rc=cuInit(0); if(rc){printf("cuInit rc=%d\n",rc);return 1;}
  int n=0; cuCount(&n); printf("devices=%d\n",n);
  if(n<2){printf("need 2 devices\n");return 0;}
  int d0,d1,can01=0,can10=0; cuGet(&d0,0); cuGet(&d1,1);
  cuCanPeer(&can01,d0,d1); cuCanPeer(&can10,d1,d0);
  printf("canAccessPeer 0->1=%d 1->0=%d\n",can01,can10);
  return 0;
}
