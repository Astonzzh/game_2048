# cython: language_level=3, boundscheck=False, wraparound=False, nonecheck=False, initializedcheck=False

from libc.stdint cimport uint64_t

# 1. Declare raw C arrays for the lookup tables (65536 = 2^16 combinations)
cdef uint64_t state_map_c[65536]
cdef uint64_t reverse_state_map_c[65536]

# We use 'double' because Expectimax calculates floating-point probabilities
cdef double row1_map_c[65536]
cdef double row2_map_c[65536]
cdef double row3_map_c[65536]
cdef double row4_map_c[65536]

# 2. Function to load your Python lists into the fast C arrays
def init_tables(list py_state_map, list py_reverse_state_map):
    cdef int i
    for i in range(65536):
        state_map_c[i] = py_state_map[i]
        reverse_state_map_c[i] = py_reverse_state_map[i]


def init_score_tables(list py_row1, list py_row2, list py_row3, list py_row4):
    cdef int i
    for i in range(65536):
        row1_map_c[i] = py_row1[i]
        row2_map_c[i] = py_row2[i]
        row3_map_c[i] = py_row3[i]
        row4_map_c[i] = py_row4[i]


# --- CORE MOVEMENT --- 
cdef inline uint64_t transpose(uint64_t board) nogil:
    cdef uint64_t keep, swap_down, swap_up, swap_1, swap_2
    keep = board & 0xF0F00F0FF0F00F0FULL
    swap_down = board & 0x0F0F00000F0F0000ULL
    swap_up = board & 0x0000F0F00000F0F0ULL
    swap_1 = keep | (swap_down >> 12) | (swap_up << 12)
    keep = swap_1 & 0xFF00FF0000FF00FFULL
    swap_down = swap_1 & 0x00FF00FF00000000ULL
    swap_up = swap_1 & 0x00000000FF00FF00ULL
    swap_2 = keep | (swap_down >> 24) | (swap_up << 24)
    return swap_2

cdef inline uint64_t move_left(uint64_t board) nogil:
    cdef uint64_t row4 = state_map_c[board & 0xFFFFULL]
    cdef uint64_t row3 = state_map_c[(board >> 16) & 0xFFFFULL]
    cdef uint64_t row2 = state_map_c[(board >> 32) & 0xFFFFULL]
    cdef uint64_t row1 = state_map_c[(board >> 48) & 0xFFFFULL]
    return (row1 << 48) | (row2 << 32) | (row3 << 16) | row4

cdef inline uint64_t move_right(uint64_t board) nogil:
    cdef uint64_t row4 = reverse_state_map_c[board & 0xFFFFULL]
    cdef uint64_t row3 = reverse_state_map_c[(board >> 16) & 0xFFFFULL]
    cdef uint64_t row2 = reverse_state_map_c[(board >> 32) & 0xFFFFULL]
    cdef uint64_t row1 = reverse_state_map_c[(board >> 48) & 0xFFFFULL]
    return (row1 << 48) | (row2 << 32) | (row3 << 16) | row4

cdef inline uint64_t move_up(uint64_t board) nogil:
    return transpose(move_left(transpose(board)))

cdef inline uint64_t move_down(uint64_t board) nogil:
    return transpose(move_right(transpose(board)))

cpdef uint64_t benchmark_move_left(uint64_t board, int iterations) nogil:
    cdef int i
    cdef uint64_t new_board
    cdef uint64_t dummy = 0
    
    # This loop now runs at pure C speed, with NO Python overhead
    for i in range(iterations):
        new_board = move_left(board)
        dummy ^= new_board  # Prevents the C compiler from optimizing the loop away
        
    return dummy


# --- SCORING HELPERS ---
cdef inline uint64_t flip_horizontal(uint64_t board) nogil:
    cdef uint64_t col1 = board & 0xF000F000F000F000ULL
    cdef uint64_t col2 = board & 0x0F000F000F000F00ULL
    cdef uint64_t col3 = board & 0x00F000F000F000F0ULL
    cdef uint64_t col4 = board & 0x000F000F000F000FULL
    return (col1 >> 12) | (col2 >> 4) | (col3 << 4) | (col4 << 12)

cdef inline uint64_t flip_vertically(uint64_t board) nogil:
    cdef uint64_t row1 = board & 0xFFFF000000000000ULL
    cdef uint64_t row2 = board & 0x0000FFFF00000000ULL
    cdef uint64_t row3 = board & 0x00000000FFFF0000ULL
    cdef uint64_t row4 = board & 0x000000000000FFFFULL
    return (row4 << 48) | (row3 << 16) | (row2 >> 16) | (row1 >> 48)

cdef inline double board_score(uint64_t board) nogil:
    cdef uint64_t row1 = board >> 48
    cdef uint64_t row2 = (board >> 32) & 0xFFFFULL
    cdef uint64_t row3 = (board >> 16) & 0xFFFFULL
    cdef uint64_t row4 = board & 0xFFFFULL
    return row1_map_c[row1] + row2_map_c[row2] + row3_map_c[row3] + row4_map_c[row4]

cdef double monotonic_score(uint64_t board) nogil:
    # Unrolled list comprehension to avoid Python objects entirely
    cdef double max_s = board_score(board)
    cdef double s
    cdef uint64_t board_t = transpose(board)
    
    s = board_score(flip_horizontal(board))
    if s > max_s: max_s = s
    s = board_score(flip_vertically(board))
    if s > max_s: max_s = s
    s = board_score(flip_horizontal(flip_vertically(board)))
    if s > max_s: max_s = s
    
    s = board_score(board_t)
    if s > max_s: max_s = s
    s = board_score(flip_horizontal(board_t))
    if s > max_s: max_s = s
    s = board_score(flip_vertically(board_t))
    if s > max_s: max_s = s
    s = board_score(flip_horizontal(flip_vertically(board_t)))
    if s > max_s: max_s = s
    
    return max_s


# Ultra-fast pure C bit counting
cdef inline int popcount(uint64_t x) nogil:
    x = x - ((x >> 1) & 0x5555555555555555ULL)
    x = (x & 0x3333333333333333ULL) + ((x >> 2) & 0x3333333333333333ULL)
    x = (x + (x >> 4)) & 0x0f0f0f0f0f0f0f0fULL
    return (x * 0x0101010101010101ULL) >> 56


# --- EXPECTIMAX ENGINE ---
# We declare get_expected_score up top so get_max_action_scores can see it
# cdef double get_expected_score(uint64_t board, int depth, int max_depth) nogil


cdef double get_max_action_scores(uint64_t board, int depth, int max_depth, int* out_action) nogil:
    cdef uint64_t new_boards[4]
    new_boards[0] = move_left(board)
    new_boards[1] = move_right(board)
    new_boards[2] = move_up(board)
    new_boards[3] = move_down(board)
    
    cdef int valid_moves = 0
    cdef double max_scores = -1.0
    cdef int max_action = -1
    cdef double score
    cdef int i
    
    for i in range(4):
        if new_boards[i] != board:
            valid_moves += 1
            score = get_expected_score(new_boards[i], depth, max_depth)
            if score > max_scores:
                max_scores = score
                max_action = i
                
    if valid_moves == 0:
        if out_action != NULL:
            out_action[0] = -1
        return 0.0 # Game Over punishment
        
    if out_action != NULL:
        out_action[0] = max_action
        
    return max_scores


cdef double get_expected_score(uint64_t board, int depth, int max_depth) nogil:
    if depth > max_depth:
        return monotonic_score(board)
        
    cdef uint64_t occupy_sign = board | (board >> 1) | (board >> 2) | (board >> 3)
    occupy_sign = ~occupy_sign
    occupy_sign &= 0x1111111111111111ULL
    
    cdef int num_empty = popcount(occupy_sign)
    if num_empty == 0:
        return 0.0
        
    cdef double base_prob = 1.0 / num_empty
    cdef double total_scores = 0.0
    cdef int i
    cdef uint64_t lowest_bit
    cdef uint64_t new_board
    
    # Pre-calculate boolean for speed
    cdef int evaluate_4_tile = 1 if depth <= 2 else 0
    
    # We fold your possible_gen directly into this loop to avoid Python lists completely
    for i in range(num_empty):
        lowest_bit = occupy_sign & -occupy_sign
        
        # Spawn a '2' tile (value 1)
        new_board = board | (lowest_bit * 1ULL)
        if evaluate_4_tile:
            total_scores += get_max_action_scores(new_board, depth + 1, max_depth, NULL) * (0.9 * base_prob)
        else:
            total_scores += get_max_action_scores(new_board, depth + 1, max_depth, NULL) * (1.0 * base_prob)
            
        # Spawn a '4' tile (value 2)
        if evaluate_4_tile:
            new_board = board | (lowest_bit * 2ULL)
            total_scores += get_max_action_scores(new_board, depth + 1, max_depth, NULL) * (0.1 * base_prob)
            
        occupy_sign &= occupy_sign - 1
        
    return total_scores


# --- PYTHON INTERFACE BINDING ---
cpdef tuple search_best_move(uint64_t board, int max_depth):
    cdef int best_action = -1
    cdef double expected_score = get_max_action_scores(board, 0, max_depth, &best_action)
    return best_action, expected_score