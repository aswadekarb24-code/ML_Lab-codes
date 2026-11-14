#include<stdio.h>
#include<omp.h>

int main(){
    int i=0, x=0;

    #pragma omp parallel shared(i)
    {
        if(omp_get_thread_num()==0){
            printf("Producer thread %d adding data\n",omp_get_thread_num());
            i+=10;
            x=1;
        } 
        else{
            while(x==0){}

            #pragma omp critical
            {
                if(i>0){
                    printf("Consumed by thread %d\n", omp_get_thread_num());
                    i-=5;
                }
            }
        }
    }
    return 0;
}