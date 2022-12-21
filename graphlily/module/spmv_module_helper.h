#include <vector>
#include <iostream>
#include "graphlily/global.h"

using std::cout;
using std::vector;
using std::min;
using std::max;

#ifndef SPARSE_HELPER
#define SPARSE_HELPER

namespace graphlily {

//----------------------------------------------------------------
// GraphLily ALUs
//----------------------------------------------------------------

// floating-point saturation addition
inline float float_sat_add(float a, float b) {
    float x = a + b;
    if (a >= FLOAT_INF) return FLOAT_INF;
    if (b >= FLOAT_INF) return FLOAT_INF;
    if (x >= FLOAT_INF) return FLOAT_INF;
    return x;
}

inline float pe_float_mul_alu(float a, float b, semiring_t op) {
    switch (op) {
        case 0:
            return a * b;
        case 1:
            return a && b;
        case 2:
            return float_sat_add(a, b);
        default: // use arithmetic semiring as the default
            return a * b;
    }
}

inline float pe_float_add_alu(float a, float b, semiring_t op) {
    switch (op) {
        case 0:
            return a + b; // TODO: sat_add
        case 1:
            return a || b;
        case 2:
            return std::min(a, b);
        default: // use arithmetic semiring as the default
            return a + b;
    }
}

struct edge{
    int col;
    int row;
    float attr;

    edge(int d = -1, int s = -1, float v = 0.0): col(d), row(s), attr(v) {}

    edge& operator=(const edge& rhs) {
        col = rhs.col;
        row = rhs.row;
        attr = rhs.attr;
        return *this;
    }
};

inline void cpu_spmv_CSR(const int M,
                  const int K,
                  const int NNZ,
                  const float ALPHA,
                  const vector<uint32_t> & CSRRowPtr,
                  const vector<uint32_t> & CSRColIndex,
                  const vector<float> & CSRVal,
                  const vector<float> & vec_X,
                  const float BETA,
                  vector<float> & vec_Y,
                  const semiring_t semi_op,
                  const unsigned semi_zero,
                  const mask_t mask_type,
                  const vector<float> & vec_MK) {
    // A: sparse matrix, M x K
    // X: dense vector, K x 1
    // Y: dense vecyor, M x 1
    // output vec_Y = ALPHA * mat_A * vec_X + BETA * vec_Y
    // dense matrices: column major

    for (int i = 0; i < M; ++i) {
        float psum = semi_zero;
        for (int j = CSRRowPtr[i]; j < CSRRowPtr[i+1]; ++j) {
            psum = pe_float_add_alu(psum, pe_float_mul_alu(CSRVal[j], vec_X[CSRColIndex[j]], semi_op), semi_op);
        }
        vec_Y[i] = ALPHA * psum + BETA * vec_Y[i];
    }

    if (mask_type == WRITETOZERO) {
        for (size_t i = 0; i < M; i++) {
            if (vec_MK[i] != 0) {
                vec_Y[i] = semi_zero;
            }
        }
    } else if (mask_type == WRITETOONE) {
        for (size_t i = 0; i < M; i++) {
            if (vec_MK[i] == 0) {
                vec_Y[i] = semi_zero;
            }
        }
    }
}

inline void generate_edge_list_for_one_PE(const vector<edge> & tmp_edge_list,
                                   vector<edge> & edge_list,
                                   const int base_col_index,
                                   const int i_start,
                                   const int NUM_Row,
                                   const int NUM_PE,
                                   const int DEP_DIST_LOAD_STORE = 10){

    edge e_empty = {-1, -1, 0.0};
    //vector<edge> scheduled_edges(NUM_Row);
    //std::fill(scheduled_edges.begin(), scheduled_edges.end(), e_empty);
    vector<edge> scheduled_edges;

    //const int DEP_DIST_LOAD_STORE = 7;

    vector<int> cycles_rows(NUM_Row, -DEP_DIST_LOAD_STORE);
    int e_dst, e_src;
    float e_attr;
    for (unsigned int pp = 0; pp < tmp_edge_list.size(); ++pp) {
        e_src = tmp_edge_list[pp].col - base_col_index;
        e_dst = tmp_edge_list[pp].row / 2 / NUM_PE; //e_dst = tmp_edge_list[pp].row / NUM_PE;
        e_attr = tmp_edge_list[pp].attr;
        auto cycle = cycles_rows[e_dst] + DEP_DIST_LOAD_STORE;

        bool taken = true;
        while (taken){
            if (cycle >= ((int)scheduled_edges.size()) ) {
                scheduled_edges.resize(cycle + 1, e_empty);
            }
            auto e = scheduled_edges[cycle];
            if (e.row != -1)
                cycle++;
            else
                taken = false;
        }
        scheduled_edges[cycle].col = e_src;
        scheduled_edges[cycle].row = e_dst * 2 + (tmp_edge_list[pp].row % 2); //scheduled_edges[cycle].row = e_dst;
        scheduled_edges[cycle].attr = e_attr;
        cycles_rows[e_dst] = cycle;
    }

    int scheduled_edges_size = scheduled_edges.size();
    if (scheduled_edges_size > 0) {
        //edge_list.resize(i_start + scheduled_edges_size + DEP_DIST_LOAD_STORE - 1, e_empty);
        edge_list.resize(i_start + scheduled_edges_size, e_empty);
        for (int i = 0; i < scheduled_edges_size; ++i) {
            edge_list[i + i_start] = scheduled_edges[i];
        }
    }
}


inline void generate_edge_list_for_all_PEs(const vector<uint32_t> & CSCColPtr,
                                    const vector<uint32_t> & CSCRowIndex,
                                    const vector<float> & CSCVal,
                                    const int NUM_PE,
                                    const int NUM_ROW,
                                    const int NUM_COLUMN,
                                    const int WINDOE_SIZE,
                                    vector<vector<edge> > & edge_list_pes,
                                    vector<int> & edge_list_ptr,
                                    const int DEP_DIST_LOAD_STORE = 10) {
    edge_list_pes.resize(NUM_PE);
    edge_list_ptr.resize((NUM_COLUMN + WINDOE_SIZE - 1) / WINDOE_SIZE + 1, 0);

    vector<vector<edge> > tmp_edge_list_pes(NUM_PE);
    for (int i = 0; i < (NUM_COLUMN + WINDOE_SIZE - 1) / WINDOE_SIZE; ++i) {
        for (int p = 0; p < NUM_PE; ++p) {
            tmp_edge_list_pes[p].resize(0);
        }

        //fill tmp_edge_lsit_pes
        for (int col =  WINDOE_SIZE * i; col < min(WINDOE_SIZE * (i + 1), NUM_COLUMN); ++col) {
            for (int j = CSCColPtr[col]; j < CSCColPtr[col+1]; ++j) {
                int p = (CSCRowIndex[j] / 2) % NUM_PE; //int p = CSCRowIndex[j] % NUM_PE;
                int pos = tmp_edge_list_pes[p].size();
                tmp_edge_list_pes[p].resize(pos + 1);
                tmp_edge_list_pes[p][pos] = edge(col, CSCRowIndex[j], CSCVal[j]);
            }
        }

        //form the scheduled edge list for each PE
        for (int p = 0; p < NUM_PE; ++p) {
            int i_start = edge_list_pes[p].size();
            int base_col_index = i * WINDOE_SIZE;
            generate_edge_list_for_one_PE(tmp_edge_list_pes[p],
                                          edge_list_pes[p],
                                          base_col_index,
                                          i_start,
                                          NUM_ROW,
                                          NUM_PE,
                                          DEP_DIST_LOAD_STORE);
        }

        //insert bubules to align edge list
        int max_len = 0;
        for (int p = 0; p < NUM_PE; ++p) {
            max_len = max((int) edge_list_pes[p].size(), max_len);
        }
        for (int p = 0; p < NUM_PE; ++p) {
            edge_list_pes[p].resize(max_len, edge(-1,-1,0.0));
        }

        //pointer
        edge_list_ptr[i+1] = max_len;
    }

}

inline void edge_list_64bit(const vector<vector<edge> > & edge_list_pes,
                     const vector<int> & edge_list_ptr,
                     vector<vector<unsigned long, tapa::aligned_allocator<unsigned long> > > & sparse_A_fpga_vec,
                     const int NUM_CH_SPARSE = 8) {

    int sparse_A_fpga_column_size = 8 * edge_list_ptr[edge_list_ptr.size()-1] * 4 / 4;
    int sparse_A_fpga_chunk_size = ((sparse_A_fpga_column_size + 511)/512) * 512;

    for (int cc = 0; cc < NUM_CH_SPARSE; ++cc) {
        sparse_A_fpga_vec[cc].resize(sparse_A_fpga_chunk_size, 0);
    }

    // col(12 bits) + row (20 bits) + value (32 bits)
    // ->
    // col(14 bits) + row (18 bits) + value (32 bits)
    for (int i = 0; i < edge_list_ptr[edge_list_ptr.size()-1]; ++i) {
        for (int cc = 0; cc < NUM_CH_SPARSE; ++cc) {
            for (int j = 0; j < 8; ++j) {
                edge e = edge_list_pes[j + cc * 8][i];
                unsigned long x = 0;
                if (e.row == -1) {
                    x = 0x3FFFF; //0xFFFFF; //x = 0x3FFFFF;
                    x = x << 32;
                } else {
                    unsigned long x_col = e.col;
                    x_col = (x_col & 0x3FFF) << (32 + 18); // x_col = (x_col & 0xFFF) << (32 + 20); //x_col = (x_col & 0x3FF) << (32 + 22);
                    unsigned long x_row = e.row;
                    x_row = (x_row & 0x3FFFF) << 32; //x_row = (x_row & 0xFFFFF) << 32; //x_row = (x_row & 0x3FFFFF) << 32;

                    float x_float = e.attr;
                    //float x_float = 1.0;
                    unsigned int x_float_in_int = *((unsigned int*)(&x_float));
                    unsigned long x_float_val_64 = ((unsigned long) x_float_in_int);
                    x_float_val_64 = x_float_val_64 & 0xFFFFFFFF;

                    x = x_col | x_row | x_float_val_64;
                }
                if (NUM_CH_SPARSE != 16 && NUM_CH_SPARSE != 24) {
                    cout << "UPDATE me\n";
                    exit(1);
                }else if (NUM_CH_SPARSE == 16) {
                    int pe_idx = j + cc * 8;
                    //ch= 0: pe  0(  0,   1) pe 16( 32,  33) pe 32( 64,  65) pe 48( 96,  97) pe 64(128, 129) pe 80(160, 161) pe 96(192, 193) pe112(224, 225)
                    //ch= 1: pe  8( 16,  17) pe 24( 48,  49) pe 40( 80,  81) pe 56(112, 113) pe 72(144, 145) pe 88(176, 177) pe104(208, 209) pe120(240, 241)
                    //ch= 2: pe  1(  2,   3) pe 17( 34,  35) pe 33( 66,  67) pe 49( 98,  99) pe 65(130, 131) pe 81(162, 163) pe 97(194, 195) pe113(226, 227)
                    //ch= 3: pe  9( 18,  19) pe 25( 50,  51) pe 41( 82,  83) pe 57(114, 115) pe 73(146, 147) pe 89(178, 179) pe105(210, 211) pe121(242, 243)
                    //ch= 4: pe  2(  4,   5) pe 18( 36,  37) pe 34( 68,  69) pe 50(100, 101) pe 66(132, 133) pe 82(164, 165) pe 98(196, 197) pe114(228, 229)
                    //ch= 5: pe 10( 20,  21) pe 26( 52,  53) pe 42( 84,  85) pe 58(116, 117) pe 74(148, 149) pe 90(180, 181) pe106(212, 213) pe122(244, 245)
                    //ch= 6: pe  3(  6,   7) pe 19( 38,  39) pe 35( 70,  71) pe 51(102, 103) pe 67(134, 135) pe 83(166, 167) pe 99(198, 199) pe115(230, 231)
                    //ch= 7: pe 11( 22,  23) pe 27( 54,  55) pe 43( 86,  87) pe 59(118, 119) pe 75(150, 151) pe 91(182, 183) pe107(214, 215) pe123(246, 247)
                    //ch= 8: pe  4(  8,   9) pe 20( 40,  41) pe 36( 72,  73) pe 52(104, 105) pe 68(136, 137) pe 84(168, 169) pe100(200, 201) pe116(232, 233)
                    //ch= 9: pe 12( 24,  25) pe 28( 56,  57) pe 44( 88,  89) pe 60(120, 121) pe 76(152, 153) pe 92(184, 185) pe108(216, 217) pe124(248, 249)
                    //ch=10: pe  5( 10,  11) pe 21( 42,  43) pe 37( 74,  75) pe 53(106, 107) pe 69(138, 139) pe 85(170, 171) pe101(202, 203) pe117(234, 235)
                    //ch=11: pe 13( 26,  27) pe 29( 58,  59) pe 45( 90,  91) pe 61(122, 123) pe 77(154, 155) pe 93(186, 187) pe109(218, 219) pe125(250, 251)
                    //ch=12: pe  6( 12,  13) pe 22( 44,  45) pe 38( 76,  77) pe 54(108, 109) pe 70(140, 141) pe 86(172, 173) pe102(204, 205) pe118(236, 237)
                    //ch=13: pe 14( 28,  29) pe 30( 60,  61) pe 46( 92,  93) pe 62(124, 125) pe 78(156, 157) pe 94(188, 189) pe110(220, 221) pe126(252, 253)
                    //ch=14: pe  7( 14,  15) pe 23( 46,  47) pe 39( 78,  79) pe 55(110, 111) pe 71(142, 143) pe 87(174, 175) pe103(206, 207) pe119(238, 239)
                    //ch=15: pe 15( 30,  31) pe 31( 62,  63) pe 47( 94,  95) pe 63(126, 127) pe 79(158, 159) pe 95(190, 191) pe111(222, 223) pe127(254, 255)

                    int pix_m16 = pe_idx % 16;
                    sparse_A_fpga_vec[(pix_m16 % 8) * 2 + pix_m16 / 8][(pe_idx % 128) / 16 + i * 8] = x;
                }else if (NUM_CH_SPARSE == 24) {
                    int pe_idx = j + cc * 8;
                    // ch= 0: pe  0(  0,   1) pe 24( 48,  49) pe 48( 96,  97) pe 72(144, 145) pe 96(192, 193) pe120(240, 241) pe144(288, 289) pe168(336, 337)
                    // ch= 1: pe  8( 16,  17) pe 32( 64,  65) pe 56(112, 113) pe 80(160, 161) pe104(208, 209) pe128(256, 257) pe152(304, 305) pe176(352, 353)
                    // ch= 2: pe 16( 32,  33) pe 40( 80,  81) pe 64(128, 129) pe 88(176, 177) pe112(224, 225) pe136(272, 273) pe160(320, 321) pe184(368, 369)
                    // ch= 3: pe  1(  2,   3) pe 25( 50,  51) pe 49( 98,  99) pe 73(146, 147) pe 97(194, 195) pe121(242, 243) pe145(290, 291) pe169(338, 339)
                    // ch= 4: pe  9( 18,  19) pe 33( 66,  67) pe 57(114, 115) pe 81(162, 163) pe105(210, 211) pe129(258, 259) pe153(306, 307) pe177(354, 355)
                    // ch= 5: pe 17( 34,  35) pe 41( 82,  83) pe 65(130, 131) pe 89(178, 179) pe113(226, 227) pe137(274, 275) pe161(322, 323) pe185(370, 371)
                    // ch= 6: pe  2(  4,   5) pe 26( 52,  53) pe 50(100, 101) pe 74(148, 149) pe 98(196, 197) pe122(244, 245) pe146(292, 293) pe170(340, 341)
                    // ch= 7: pe 10( 20,  21) pe 34( 68,  69) pe 58(116, 117) pe 82(164, 165) pe106(212, 213) pe130(260, 261) pe154(308, 309) pe178(356, 357)
                    // ch= 8: pe 18( 36,  37) pe 42( 84,  85) pe 66(132, 133) pe 90(180, 181) pe114(228, 229) pe138(276, 277) pe162(324, 325) pe186(372, 373)
                    // ch= 9: pe  3(  6,   7) pe 27( 54,  55) pe 51(102, 103) pe 75(150, 151) pe 99(198, 199) pe123(246, 247) pe147(294, 295) pe171(342, 343)
                    // ch=10: pe 11( 22,  23) pe 35( 70,  71) pe 59(118, 119) pe 83(166, 167) pe107(214, 215) pe131(262, 263) pe155(310, 311) pe179(358, 359)
                    // ch=11: pe 19( 38,  39) pe 43( 86,  87) pe 67(134, 135) pe 91(182, 183) pe115(230, 231) pe139(278, 279) pe163(326, 327) pe187(374, 375)
                    // ch=12: pe  4(  8,   9) pe 28( 56,  57) pe 52(104, 105) pe 76(152, 153) pe100(200, 201) pe124(248, 249) pe148(296, 297) pe172(344, 345)
                    // ch=13: pe 12( 24,  25) pe 36( 72,  73) pe 60(120, 121) pe 84(168, 169) pe108(216, 217) pe132(264, 265) pe156(312, 313) pe180(360, 361)
                    // ch=14: pe 20( 40,  41) pe 44( 88,  89) pe 68(136, 137) pe 92(184, 185) pe116(232, 233) pe140(280, 281) pe164(328, 329) pe188(376, 377)
                    // ch=15: pe  5( 10,  11) pe 29( 58,  59) pe 53(106, 107) pe 77(154, 155) pe101(202, 203) pe125(250, 251) pe149(298, 299) pe173(346, 347)
                    // ch=16: pe 13( 26,  27) pe 37( 74,  75) pe 61(122, 123) pe 85(170, 171) pe109(218, 219) pe133(266, 267) pe157(314, 315) pe181(362, 363)
                    // ch=17: pe 21( 42,  43) pe 45( 90,  91) pe 69(138, 139) pe 93(186, 187) pe117(234, 235) pe141(282, 283) pe165(330, 331) pe189(378, 379)
                    // ch=18: pe  6( 12,  13) pe 30( 60,  61) pe 54(108, 109) pe 78(156, 157) pe102(204, 205) pe126(252, 253) pe150(300, 301) pe174(348, 349)
                    // ch=19: pe 14( 28,  29) pe 38( 76,  77) pe 62(124, 125) pe 86(172, 173) pe110(220, 221) pe134(268, 269) pe158(316, 317) pe182(364, 365)
                    // ch=20: pe 22( 44,  45) pe 46( 92,  93) pe 70(140, 141) pe 94(188, 189) pe118(236, 237) pe142(284, 285) pe166(332, 333) pe190(380, 381)
                    // ch=21: pe  7( 14,  15) pe 31( 62,  63) pe 55(110, 111) pe 79(158, 159) pe103(206, 207) pe127(254, 255) pe151(302, 303) pe175(350, 351)
                    // ch=22: pe 15( 30,  31) pe 39( 78,  79) pe 63(126, 127) pe 87(174, 175) pe111(222, 223) pe135(270, 271) pe159(318, 319) pe183(366, 367)
                    // ch=23: pe 23( 46,  47) pe 47( 94,  95) pe 71(142, 143) pe 95(190, 191) pe119(238, 239) pe143(286, 287) pe167(334, 335) pe191(382, 383)

                    int pix_m24 = pe_idx % 24;
                    sparse_A_fpga_vec[(pix_m24 % 8) * 3 + pix_m24 / 8][(pe_idx % 192) / 24 + i * 8] = x;
                }
            }
        }
    }
}

}

#endif
