#ifndef METRICS_H
#define METRICS_H

double compute_ari(const int* y_true, const int* y_pred, int n);
double compute_nmi(const int* y_true, const int* y_pred, int n);
double compute_acc(const int* y_true, const int* y_pred, int n);

#endif /* METRICS_H */
