#include<stdio.h>
#include<omp.h>

int main(){
    int n_threads, tid;

    #pragma omp parallel private(n_threads, tid)
    {
        tid=omp_get_thread_num();
        printf("Hello World form thread=%d\n", tid);

        if(tid==0){
            n_threads=omp_get_num_threads();
            printf("Number of threads=%d\n", n_threads);
        }
    }
    return 0;
}