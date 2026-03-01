# cython: language_level=3, boundscheck=False, wraparound=False, nonecheck=False, initializedcheck=False

from libc.stdint cimport uint64_t

# 1. Declare raw C arrays for the lookup tables (65536 = 2^16 combinations)
cdef uint64_t state_map_c[65536]
cdef uint64_t reverse_state_map_c[65536]

# 2. Function to load your Python lists into the fast C arrays
def init_tables(list py_state_map, list py_reverse_state_map):
    cdef int i
    for i in range(65536):
        state_map_c[i] = py_state_map[i]
        reverse_state_map_c[i] = py_reverse_state_map[i]


cdef inline uint64_t transpose(uint64_t board) nogil:
    cdef uint64_t keep, swap_down, swap_up, swap_1, swap_2
    
    # Notice the ULL at the end of every hex literal
    keep = board & 0xF0F00F0FF0F00F0FULL
    swap_down = board & 0x0F0F00000F0F0000ULL
    swap_up = board & 0x0000F0F00000F0F0ULL
    swap_1 = keep | (swap_down >> 12) | (swap_up << 12)

    keep = swap_1 & 0xFF00FF0000FF00FFULL
    swap_down = swap_1 & 0x00FF00FF00000000ULL
    swap_up = swap_1 & 0x00000000FF00FF00ULL
    swap_2 = keep | (swap_down >> 24) | (swap_up << 24)

    return swap_2


# 4. Use 'cpdef' for functions you want to call from your main Python script
cpdef uint64_t move_left(uint64_t board):
    cdef uint64_t row4 = state_map_c[board & 0xFFFF]
    cdef uint64_t row3 = state_map_c[(board >> 16) & 0xFFFF]
    cdef uint64_t row2 = state_map_c[(board >> 32) & 0xFFFF]
    cdef uint64_t row1 = state_map_c[(board >> 48) & 0xFFFF]
    return (row1 << 48) | (row2 << 32) | (row3 << 16) | row4

cpdef uint64_t move_right(uint64_t board):
    cdef uint64_t row4 = reverse_state_map_c[board & 0xFFFF]
    cdef uint64_t row3 = reverse_state_map_c[(board >> 16) & 0xFFFF]
    cdef uint64_t row2 = reverse_state_map_c[(board >> 32) & 0xFFFF]
    cdef uint64_t row1 = reverse_state_map_c[(board >> 48) & 0xFFFF]
    return (row1 << 48) | (row2 << 32) | (row3 << 16) | row4

cpdef uint64_t move_up(uint64_t board):
    return transpose(move_left(transpose(board)))

cpdef uint64_t move_down(uint64_t board):
    return transpose(move_right(transpose(board)))

cpdef uint64_t benchmark_move_left(uint64_t board, int iterations):
    cdef int i
    cdef uint64_t new_board
    cdef uint64_t dummy = 0
    
    # This loop now runs at pure C speed, with NO Python overhead
    for i in range(iterations):
        new_board = move_left(board)
        dummy ^= new_board  # Prevents the C compiler from optimizing the loop away
        
    return dummy