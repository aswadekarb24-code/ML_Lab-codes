#include<stdio.h>
#include<omp.h>

int isPrime(int x){
    for(int i=2; i<x; i++){
        if(x%i==0 && i!=x){
            return 0;
        }
    }
    return 1;
}

int main(){
    int n, idx=0, temp;
    int arr[10000];

    printf("Enter the number N: \n");
    scanf("%d",&n);

    #pragma omp parallel for reduction(+:idx)
    for(temp=2; temp<=n; temp++){
        if(isPrime(temp)){
            arr[idx]=temp;
            idx++;
        }
    }
    printf("Number of prime numbers between 2 and %d: %d\n",n,idx);
    return 0;
}