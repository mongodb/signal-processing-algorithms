#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>

/* public functions */
bool t_stat_values(double *distances, double * values, int length);
bool calculate_distance_matrix(double *series, double *distances, int length);
double calculate_t(double cross_term, double x_term, double y_term, int x_len, int y_len);
double square_sum(double * distances, int length, int row_start, int row_end, int column_start, int column_end);

/**
 * Calculate the pairwise distances within the input series.
 *
 * @param distances The NxN double array to calculate the distances for.
 * @param series The series of values to calculate a distance matrix for.
 * @param length The length of the series of values.
 */
bool calculate_distance_matrix(double *series, double *distances, int length){
    int i;
    int j;

    for(i=0;i<length;i++){
        for(j=0;j<length;j++){
            distances[i * length + j] = fabs(series[i] - series[j]);
        }
    }
    return true;
}

/**
 * Calculate a single t value from the terms.
 *
 * @param cross_term The sum of the distances across partitions.
 * @param x_term The sum of the distances within the X partition.
 * @param term3 The sum of the distances within the Y partition.
 * @param x_len The length/size of the X partition.
 * @param y_len The length/size of the Y partition.
 */
double calculate_t(double cross_term, double x_term, double y_term, int x_len, int y_len) {
    double cross_term_reg;
    double x_term_reg;
    double y_term_reg;

    if(x_len < 1 || y_len < 1) {
        cross_term_reg = 0;
    } else {
        cross_term_reg = cross_term * (2.0 / (x_len * y_len));
    }

    if(x_len < 1) {
        x_term_reg = 0;
    } else {
        x_term_reg = x_term / (x_len * x_len);
    }

    if(y_len < 1) {
        y_term_reg = 0;
    } else {
        y_term_reg = y_term / (y_len * y_len);
    }

    double factor = ((double) x_len * y_len / (x_len + y_len));

    return  factor * (cross_term_reg - x_term_reg - y_term_reg);
}

/**
 * Calculate the sum of terms in a NxN distance matrix within
 * the square [row_start, row_end) x [column_start, column_end).
 *
 * @param distances The NxN distance matrix.
 * @param length The length of one dimension of the distance matrix, i.e. the integer N.
 * @param row_start Index of the row where the square begins (inclusive).
 * @param row_end Index of the row where the square ends (exclusive).
 * @param column_start Index of the column where the square begins (inclusive).
 * @param column_end Index of the column where the square ends (exclusive).
 */
double square_sum(double * distances, int length, int row_start, int row_end, int column_start, int column_end) {
    int row;
    int column;
    double sum = 0.0;
    for(row=row_start;row<row_end;row++) {
        for(column=column_start;column<column_end;column++) {
            sum = sum + distances[row * length + column];
        }
    }
    return sum;
}

/**
 * Calculate all the t values for the input series and store the results in the values
 * array.
 *
 * @param distances The NxN distance matrix for the series.
 * @param t_stat_values The array to store the t values in.
 * @param length Length of the series, i.e. N.
 * @return true for success.
 */
bool t_stat_values(double * distances, double * t_stat_values, int length){
    // We will partition our signal into:
    // X = {Xi; 0 <= i < tau}
    // Y = {Yj; tau <= j < len(signal) }
    // and look for argmax(tau)Q(tau)

    // sum |Xi - Yj| for i < tau <= j
    double cross_term = 0;
    // sum |Xi - Xj| for i < j < tau
    double x_term = 0;
    // sum |Yi - Yj| for tau <= i < j
    double y_term = 0;

    int row;
    for(row=0;row<length;row++) {
        y_term += square_sum(distances, length, row, row+1, row, length);
    }

    int tau;
    for(tau=0;tau<length;tau++) {
        t_stat_values[tau] = calculate_t(cross_term, 2*x_term, 2*y_term, tau, length - tau);

        double column_delta = square_sum(distances, length, 0, tau, tau, tau+1);
        double row_delta = square_sum(distances, length, tau, tau+1, tau, length);

        cross_term = cross_term - column_delta + row_delta;
        x_term = x_term + column_delta;
        y_term = y_term - row_delta;

    }

    return true;
}
