#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <pthread.h>
#include <unistd.h>
#include <omp.h>

int len = 100;
int* a = (int*)malloc(sizeof(int)*len);
int* b = (int*)malloc(sizeof(int)*len);
int* c = (int*)malloc(sizeof(int)*len);

void set() {
    for (int i=0; i<len; i++) {
        a[i] = 1;
        b[i] = 2;
    }
}

void* add_part(void* data) {
    int* idx_ = (int*)data;
    int idx = *idx_;
    
    for (int j=0; j<1000*1000*300; j++) {
		if (j%1000 == 0) printf("iter : %d\n", j);
        for (int i=idx; i<len; i+=4) {
            c[i] = a[i] + b[i];
        }
    }
    return (void*)(NULL);
}

void add_1() {    
    clock_t start, end;
    float result;

    printf("started\n");
    start = clock();
    for (int i=0; i<len; i++)
        c[i] = a[i] + b[i];
    end = clock();

    result = (float)(end-start)/CLOCKS_PER_SEC;
    printf("add time : %f\n", result);
}

void add_2() {
    pthread_t pthread[4];
    clock_t start, end;
    float result;
    int thr_id;
    int status;

    printf("started\n");
    start = clock();
    int *data = (int*)malloc(sizeof(int));

    *data = 0;
	printf("fork to pthread0\n");
    thr_id = pthread_create(&pthread[0], NULL, add_part, (void*)(data));
    *data = 1;
	printf("fork to pthread1\n");
    thr_id = pthread_create(&pthread[1], NULL, add_part, (void*)(data));
    *data = 2;
	printf("fork to pthread2\n");
    thr_id = pthread_create(&pthread[2], NULL, add_part, (void*)(data));
    *data = 3;
	printf("fork to pthread3\n");
    thr_id = pthread_create(&pthread[3], NULL, add_part, (void*)(data));

    pthread_join(pthread[0], (void **)&status);
    pthread_join(pthread[1], (void **)&status);
    pthread_join(pthread[2], (void **)&status);
    pthread_join(pthread[3], (void **)&status);    

    end = clock();

    result = (float)(end-start)/CLOCKS_PER_SEC;
    printf("add time : %f\n", result);
}

void add_3() {    
    clock_t start, end;
    float result;
    omp_set_num_threads(4);

    printf("started\n");
    start = clock();
    #pragma omp parallel
    {
		for(int j=0; j<1000*1000*300; j++) {
			if (j%1000 == 0) printf("iter : %d\n", j);
			for(int i=0; i<len; i++)
				c[i] = a[i] + b[i];
		}
    }
    end = clock();

    result = (float)(end-start)/CLOCKS_PER_SEC;
    printf("add time : %f\n", result);
}

int main() {
    int option;
    printf("1:normal_add, 2:pthread_add, 3:pragma_add\noption: ");
    scanf("%d", &option);

    set();

    if(option == 1)
        add_1();
    else if (option == 2)
        add_2();
    else if (option == 3)
        add_3();

    return 0; 
}

