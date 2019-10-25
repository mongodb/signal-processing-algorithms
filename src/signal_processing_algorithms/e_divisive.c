#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

/* public functions */
int qhat_values(double * series, double *diffs, double * values, int length);
int calculate_diffs(double *series, double *diffs, int  length);

/* private functions */
static double calculate_qhat_value(double term1, double term2, double term3, int n, int length);
static double calculate_term1(double *diffs, int n, int  length);
static double term1_reg(double term, int n, int  length);
static double calculate_term2(double *diffs, int n, int  length);
static double term2_reg(double term, int n);
static double calculate_term3(double *diffs, int n, int  length);
static double term3_reg(double term, int n, int  length);

/**
 * Calculate the diffs from the input series.
 *
 * @param diffs The NxN double array to calculate the differences for. This 
 * only needs to be performed once.
 * @param series The array of time series data.
 * @param length The size of the time series data.
 */
int calculate_diffs(double *series, double *diffs, int  length){
    int i;
    int j;

    for(i=0;i<length;i++){
        for(j=0;j<length;j++){
            diffs[i * length + j] = fabs(series[i] - series[j]);
        }
    }
    return 0;
}

/**
 * Calculate a single qhat value from the terms.
 *
 * @param term1 The sum of the differences for term1.
 * @param term2 The sum of the differences for term2.
 * @param term3 The sum of the differences for term3.
 * @param n The current position.
 * @param length The size of the time series data.
 */
double calculate_qhat_value(double term1, double term2, double term3, int n, int length){
    int m = length - n;
    return (m * n / (m + n)) * (term1_reg(term1, n, length) - term2_reg(term2, n) - term3_reg(term3, n, length));
}

/**
 * Calculate the sum of the differences for term1.
 * @param diffs A matrix of the differences.
 * @param n The current position.
 * @param length The size of the input array.
 */
double calculate_term1(double *diffs, int n, int  length){
    double term1 = 0.0;
    int i;
    int j;
    for(i=0;i<n;i++){
        for(j=n;j<length;j++){
            term1 += diffs[i * length + j];
        }
    }
    return term1;
}

/**
 * The average term1 based on the sample. See equation 5 in https://arxiv.org/pdf/1306.4933.pdf.
 * @param term The sum of the diffs for the first term.
 * @param n The current position.
 * @param length The size of the input array.
 */
double term1_reg(double term, int n, int  length){
    int m = length - n;
    return (term * (2.0 / (m * n)));
}

/**
 * Calculate the sum of the differences for term2.
 * @param diffs A matrix of the differences.
 * @param n The current position.
 * @param length The size of the input array.
 */
double calculate_term2(double *diffs, int n, int  length){
    double term2 = 0.0;
    int i;
    int k;
    for(i=0;i<n;i++){
        for(k=i+1;k<n;k++){
            term2 += diffs[i * length + k];
        }
    }
    return term2;
}

/**
 * The average term2 based on the sample. See equation 5 in https://arxiv.org/pdf/1306.4933.pdf.
 * @param term The sum of the diffs for the second term.
 * @param n The current position.
 */
double term2_reg(double term, int n){
    return (term * (2.0 / (n * (n - 1))));
}

/**
 * Calculate the sum of the differences for term3.
 * @param diffs A matrix of the differences.
 * @param n The current position.
 * @param length The size of the input array.
 */
double calculate_term3(double *diffs, int n, int  length){
    double term3 = 0.0;
    int j;
    int k;
    for(j=n;j<length;j++){
        for(k=j+1;k<length;k++){
            term3 += diffs[j * length + k];
        }
    }
    return term3;
}

/**
 * The average term3 based on the sample. See equation 5 in https://arxiv.org/pdf/1306.4933.pdf.
 * @param term The sum of the diffs for the third term.
 * @param n The current position.
 * @param length The size of the input array.
 */
double term3_reg(double term, int n, int  length){
    int m = length - n;
    return (term * (2.0 / (m * (m - 1))));
}

/**
 * Calculate all the qhat values for the input series and store the results in the values
 * array.
 *
 * @param series The time series data.
 * @param diffs The matrix of diff data.
 * @param values The array to store the qhat values in.
 * @param length The size of the time series data.
 * @return 0 for success.
 */
int qhat_values(double * series, double * diffs, double * values, int length){
    int i;
    int n;

    double term1;
    double term2;
    double term3;

    n = 2;
    term1 = calculate_term1(diffs, n, length);
    term2 = calculate_term2(diffs, n, length);
    term3 = calculate_term3(diffs, n, length);

    values[n] = calculate_qhat_value(term1, term2, term3, n, length);

    for(n=3;n<length-2;n++){
        double row_delta = 0.0;
        double column_delta = 0.0;
        
        for(i=0; i < n -1; i++) {
            row_delta += diffs[(n - 1) * length + i];
        }

        for(i=n; i < length; i++) {
            column_delta += diffs[i * length + n - 1];
        }

        term1 = term1 - row_delta + column_delta;
        term2 = term2 + row_delta;
        term3 = term3 - column_delta;

        values[n] = calculate_qhat_value(term1, term2, term3, n, length);
    }
    return 0;
}
