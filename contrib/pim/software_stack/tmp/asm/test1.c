#include <stdio.h>
#include <stdint.h>

int main() {
	/*
	uint64_t hex_addr = 1234;
	asm volatile (
		"mov eax, %0"
		"mov BYTE PTR [rax], 1"
		: "=r" ( val )
		: "r" ( hex_addr )
		: "%eax"
		);
	*/
	int hex_addr = 1234, val ;
    asm volatile (
		"movl %0, %%eax;"
       	"movq $1, %%rax;"
       	: "=r" ( val )		/* output */
       	: "r" ( hex_addr )	/* input */
       	: "%eax"         	/* clobbered register */
    );

	printf("%d\n", val);

	return 0;
}
