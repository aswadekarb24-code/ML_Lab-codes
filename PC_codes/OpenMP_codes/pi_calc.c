#include<stdio.h>
#include<omp.h>

int main(){
    int num_steps=10000, i;
    double aux, pi, step=1.0/num_steps, x=0.0, sum=0.0;

    #pragma omp parallel private(i,x,aux) shared(sum)
    {
        #pragma omp for schedule(static)
        for(i=0; i<num_steps; i++){
            x=(i+0.5)*step;
            aux=4.0/(1.0+x*x);

            #pragma omp critical
            sum+=aux;
        }
    }
    pi=step*sum;
    printf("Value of Pi is %lf\n",pi);
    return 0;
}