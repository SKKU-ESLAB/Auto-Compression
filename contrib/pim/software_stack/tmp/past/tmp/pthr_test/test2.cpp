#include <pthread.h>
#include <iostream>

#define num_ch ((int)2)
#define num_thr_per_ch ((int)4)

typedef struct {
	int ch;
	int num;
} thr_param_t;


pthread_t thr_grp[num_ch];
thr_param_t thr_grp_param[num_ch];
pthread_t thr[num_ch * num_thr_per_ch];
thr_param_t thr_param[num_ch * num_thr_per_ch];

pthread_barrier_t thr_barrier[num_ch];
pthread_mutex_t mutex[num_ch];
pthread_mutex_t print_mutex;

static void* PrintNum(void *input_) {
	thr_param_t *input = (thr_param_t*)input_;
	int ch = input->ch;
	int num = input->num;

	pthread_mutex_lock(&print_mutex);
	for (int i=0; i<ch; i++)
		std::cout << "\t";
	std::cout << "   " << num << "\n";
	pthread_mutex_unlock(&print_mutex);

	pthread_barrier_wait(&thr_barrier[ch]);
	return (NULL);
}

static void* PrintNumChannel(void *input_) {
	thr_param_t *input = (thr_param_t*)input_;
	int ch = input->ch;
	int num = input->num;

	for (int off_i=0; off_i < 2; off_i++) {
		pthread_barrier_init(&thr_barrier[ch], NULL, 4);
		for (int off_f=0; off_f < 4; off_f++) {
			int offset = off_i*4 + off_f;
			thr_param[ch*4+off_f].ch = ch;
			thr_param[ch*4+off_f].num = offset;
			pthread_create(&thr[ch*4+off_f], NULL, PrintNum, (void*)&thr_param[ch*4+off_f]);
		}
	}
	return (NULL);
}

void try1() {
	for (int ch=0; ch < num_ch; ch++) {
		thr_grp_param[ch].ch = ch;
		thr_grp_param[ch].num = 0;
		pthread_create(&thr_grp[ch], NULL, PrintNumChannel, (void*)&thr_grp_param[ch]);
	}
}

int main() {
	std::cout << "  ch0\t  ch1\n";
	try1();
	
	for (int ch=0; ch < 2; ch++) {
		pthread_join(thr_grp[ch], NULL);
		for (int offset=0; offset < 8; offset++)
			pthread_join(thr[ch*8+offset], NULL);
	}
	std::cout << "finished!\n";
	
	return 0;
}	

