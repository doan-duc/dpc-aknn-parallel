#include "metrics.h"
#include <stdlib.h>
#include <math.h>

/* Compute the Adjusted Rand Index. */
double compute_ari(const int* y_true, const int* y_pred, int n) {
    int max_true = 0, max_pred = 0;
    for (int i = 0; i < n; i++) {
        if (y_true[i] > max_true) max_true = y_true[i];
        if (y_pred[i] > max_pred) max_pred = y_pred[i];
    }
    int R = max_true + 1, C = max_pred + 1;
    int* table   = (int*)calloc((size_t)R * (size_t)C, sizeof(int));
    int* row_sum = (int*)calloc((size_t)R, sizeof(int));
    int* col_sum = (int*)calloc((size_t)C, sizeof(int));
    for (int i = 0; i < n; i++) {
        table[y_true[i] * C + y_pred[i]]++;
        row_sum[y_true[i]]++;
        col_sum[y_pred[i]]++;
    }

    double sum_nij = 0, sum_ai = 0, sum_bj = 0;
    for (int r = 0; r < R; r++) {
        double x = row_sum[r];
        sum_ai += (x * (x - 1)) / 2.0;
        for (int c = 0; c < C; c++) {
            double nij = table[r * C + c];
            sum_nij += (nij * (nij - 1)) / 2.0;
        }
    }
    for (int c = 0; c < C; c++) {
        double y = col_sum[c];
        sum_bj += (y * (y - 1)) / 2.0;
    }
    double total    = ((double)n * (n - 1)) / 2.0;
    double expected = total > 0 ? (sum_ai * sum_bj) / total : 0.0;
    double max_idx  = 0.5 * (sum_ai + sum_bj);
    
    free(table); free(row_sum); free(col_sum);
    
    if (fabs(max_idx - expected) < 1e-12) return 1.0;
    return (sum_nij - expected) / (max_idx - expected);
}

/* Compute normalized mutual information. */
double compute_nmi(const int* y_true, const int* y_pred, int n) {
    int max_true = 0, max_pred = 0;
    for (int i = 0; i < n; i++) {
        if (y_true[i] > max_true) max_true = y_true[i];
        if (y_pred[i] > max_pred) max_pred = y_pred[i];
    }
    int R = max_true + 1, C = max_pred + 1;
    int* table   = (int*)calloc((size_t)R * (size_t)C, sizeof(int));
    int* row_sum = (int*)calloc((size_t)R, sizeof(int));
    int* col_sum = (int*)calloc((size_t)C, sizeof(int));
    for (int i = 0; i < n; i++) {
        table[y_true[i] * C + y_pred[i]]++;
        row_sum[y_true[i]]++; col_sum[y_pred[i]]++;
    }
    double mi = 0.0;
    for (int r = 0; r < R; r++) {
        for (int c = 0; c < C; c++) {
            if (table[r * C + c] == 0) continue;
            double p_rc = (double)table[r * C + c] / n;
            double p_r  = (double)row_sum[r]   / n;
            double p_c  = (double)col_sum[c]   / n;
            mi += p_rc * log(p_rc / (p_r * p_c));
        }
    }
    double h_true = 0.0, h_pred = 0.0;
    for (int r = 0; r < R; r++) if (row_sum[r] > 0) {
        double p = (double)row_sum[r] / n;
        h_true -= p * log(p);
    }
    for (int c = 0; c < C; c++) if (col_sum[c] > 0) {
        double p = (double)col_sum[c] / n;
        h_pred -= p * log(p);
    }
    free(table); free(row_sum); free(col_sum);
    double denom = 0.5 * (h_true + h_pred);
    return denom > 1e-12 ? mi / denom : 0.0;
}

/* Find a maximum-weight matching with the O(N^3) Hungarian algorithm. */
static int hungarian_max_weight(const int* weight_matrix, int N) {
    if (N == 0) return 0;
    int max_w = 0;
    for (int i = 0; i < N * N; i++) {
        if (weight_matrix[i] > max_w) max_w = weight_matrix[i];
    }
    
    int* u = (int*)calloc((size_t)(N + 1), sizeof(int));
    int* v = (int*)calloc((size_t)(N + 1), sizeof(int));
    int* p = (int*)calloc((size_t)(N + 1), sizeof(int));
    int* way = (int*)calloc((size_t)(N + 1), sizeof(int));
    int* minv = (int*)malloc((size_t)(N + 1) * sizeof(int));
    int* used = (int*)malloc((size_t)(N + 1) * sizeof(int));

    for (int i = 1; i <= N; i++) {
        p[0] = i;
        int j0 = 0;
        for (int j = 0; j <= N; j++) {
            minv[j] = 1e9;
            used[j] = 0;
        }
        
        do {
            used[j0] = 1;
            int i0 = p[j0], delta = 1e9, j1 = 0;
            for (int j = 1; j <= N; j++) {
                if (!used[j]) {
                    int cost = max_w - weight_matrix[(i0 - 1) * N + (j - 1)];
                    int cur = cost - u[i0] - v[j];
                    if (cur < minv[j]) {
                        minv[j] = cur;
                        way[j] = j0;
                    }
                    if (minv[j] < delta) {
                        delta = minv[j];
                        j1 = j;
                    }
                }
            }
            for (int j = 0; j <= N; j++) {
                if (used[j]) {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }
            j0 = j1;
        } while (p[j0] != 0);
        
        do {
            int j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
        } while (j0 != 0);
    }
    
    int ans = 0;
    for (int j = 1; j <= N; j++) {
        ans += weight_matrix[(p[j] - 1) * N + (j - 1)];
    }
    
    free(u); free(v); free(p); free(way); free(minv); free(used);
    return ans;
}

/* Compute clustering accuracy using Hungarian matching. */
double compute_acc(const int* y_true, const int* y_pred, int n) {
    int max_true = 0, max_pred = 0;
    for (int i = 0; i < n; i++) {
        if (y_true[i] > max_true) max_true = y_true[i];
        if (y_pred[i] > max_pred) max_pred = y_pred[i];
    }
    
    /* Hungarian matching requires a square weight matrix. */
    int N = (max_true > max_pred ? max_true : max_pred) + 1;
    
    int* table = (int*)calloc((size_t)(N * N), sizeof(int));
    for (int i = 0; i < n; i++) {
        table[y_true[i] * N + y_pred[i]]++;
    }
    
    int correct = hungarian_max_weight(table, N);
    free(table);
    
    return (double)correct / (double)n;
}
