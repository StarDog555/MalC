#include <stddef.h> 
#include <stdint.h> 
#include <stdlib.h> 
#include <stdio.h>

#include "../include/arena.h"

static int CheckSize(size_t Size)  
{
    if (Size == 0) 
    {
        printf("[-] Error: Enter a Valid Arena Size\n");
        return 1;
    }

    return 0;
}	


void Arena_Create(Arena *a, size_t Size)
{
    if (a == NULL) 
    {
        printf("[-] Error: A is NULL\n");
        return;
    }

    if (CheckSize(Size)) 
    {
        exit(EXIT_FAILURE);
    }

    a->Buffer = malloc(Size);

    if (a->Buffer == NULL) 
    {     
        printf("[-] Error: Malloc Failed\n");     
        exit(EXIT_FAILURE); 
    }

    a->Capacity = Size;
    a->Offset = 0;
}


void Arena_Destroy(Arena *a)
{
    if (a == NULL) 
    {
        printf("[-] Error: A is NULL\n");
        return;	   
    }
   
    free(a->Buffer);

    a->Buffer = NULL;
    a->Offset = 0;
    a->Capacity = 0;
}


void *Arena_Alloc(Arena *a, size_t Size)
{
    if (a == NULL) 
    {
        printf("[-] Error: A is NULL\n");
        return NULL;
    }
   
    if (Size > a->Capacity - a->Offset)
    {
        printf("[-] Error: Out of Memory\n");	
        return NULL;
    }

    void *ptr = a->Buffer + a->Offset;
    a->Offset += Size;

    return ptr;
}


void Arena_Reset(Arena *a) 
{
    if (a == NULL) 
    {
        printf("[-] Error: A is NULL\n");
        return;
    }
   
    a->Offset = 0;
}

void Arena_Grow(Arena *a, size_t NewSize) 
{
    if (a == NULL) 
    {
        printf("[-] Error: A is NULL\n");
        return;
    } 
   
    if (CheckSize(NewSize))
    {
        return;
    }
    
    if (NewSize < a->Offset) 
    {
        printf("[-] Error: New Size Is Smaller Than Used Memory\n");
        return;
    }

    void *new_buffer = realloc(a->Buffer, NewSize);

    if (new_buffer == NULL) 
    {    
        printf("[-] Error: Grow Failed\n");     
        return;
    }

    a->Buffer = new_buffer;
    a->Capacity = NewSize;
}


void *Arena_Alloc_Tensor(Arena *a, size_t elements, size_t element_size)
{
    if (a == NULL) 
    {
        printf("[-] Error: A is NULL\n");
        return NULL;
    } 
	
    if (element_size == 0) 
    {
        printf("[-] Error: element_size Cannot Be 0\n");   
        return NULL;
    }	
	
    if (elements > SIZE_MAX / element_size) 
    {
        printf("[-] Error: Integer Overflow\n");   
        return NULL;	   
    } 

    size_t TotalSize = elements * element_size;

    return Arena_Alloc(a, TotalSize);
}
