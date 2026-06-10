#include <stdio.h>
#include <stdint.h>

int main() {
    uint32_t eax, ebx, ecx, edx;

    printf("--- CPU Cache Topology (Hardware Level) ---\n");

    // Sub-leaf index 'i' identifies which cache we are querying
    for (int i = 0; i < 10; i++) {
        asm volatile(
            "cpuid"
            : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx) // Output
            : "a"(0x04), "c"(i)                         // Input (Leaf 4, Sub-leaf i)
        );

        int type = eax & 0x1F;
        if (type == 0) break;

        // Bits [7:5] contain the Cache Level
        int level = (eax >> 5) & 0x07;

        // Formula for Total Cache Size:
        // Size = (Ways + 1) * (Partitions + 1) * (Line_Size + 1) * (Sets + 1)
        int ways       = ((ebx >> 22) & 0x3FF) + 1;
        int partitions = ((ebx >> 12) & 0x3FF) + 1;
        int line_size  = (ebx & 0xFFF) + 1;
        int sets       = ecx + 1;

        long size_bytes = (long)ways * partitions * line_size * sets;

        const char* type_str;
        switch(type) {
            case 1: type_str = "Data"; break;
            case 2: type_str = "Instruction"; break;
            case 3: type_str = "Unified"; break;
            default: type_str = "Unknown";
        }

        printf("L%d %-12s | Size: %7ld KB | Line Size: %d bytes | Ways: %d\n", 
                level, type_str, size_bytes / 1024, line_size, ways);
    }

}