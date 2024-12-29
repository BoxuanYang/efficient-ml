#include <assert.h>
#include <pthread.h>
#include <stdio.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <cstdint>

int main(){
    unsigned char x = 0xDE;
    printf("Unsigned char in hex: 0x%X\n", x);

    unsigned char y = x & 0x0F;
    printf("Unsigned char in hex: 0x%X\n", y);

    y = x << 4;
    printf("Unsigned char in hex: 0x%X\n", y);

    y = x >> 4;
    printf("Unsigned char in hex: 0x%X\n", y);


}