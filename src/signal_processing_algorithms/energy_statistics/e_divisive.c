#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <float.h>

/* public functions */
bool t_stat_values(double *distances, double * values, int length);
bool calculate_distance_matrix(double *series, double *distances, int length);
double calculate_t(double cross_term, double x_term, double y_term, int x_len, int y_len);
double square_sum(double * distances, int length, int row_start, int row_end, int column_start, int column_end);

bool largest_q(double * distances, double * largest_q_result, int length, int min_cluster_size);
double sum_fixed_column(double * distances, int length, int row_start, int row_end, int column);
double sum_fixed_row(double * distances, int length, int row, int column_start, int column_end);

/**
 * Calculate the largest Q value and its index by using the E-Divisive Mean algorithm.
 *
 * @param distances The NxN double array that contains the distances.
 * @param largest_q_result An array that stores the result
 * array[0]: if 1, than a largest Q was computed
 * array[1]: largest Q value
 * array[2]: Index dividing the cluster (a change point candidate)
 * @param length The length of the series of values.
 * #param min_cluster_size The minimum amount of data points for a cluster.
 */
bool largest_q(double * distances, double * largest_q_result, int length, int min_cluster_size) {
    // Please see
    // src/signal_processing_algorithms/energy_statistics/energy_statistics.py:_calculate_largest_q
    // for an illustration on how the E-Divisive Mean algorithm works. This knowledge may be required
    // to understand the following calculation.
    //
    // The goal of this function is to compute all Q values as efficient as possible, therefore all kinds
    // of conditions are avoided.
    
    // Use double to avoid casting
    double x_size;
    double y_size;

    double x = 0.0;
    double y = 0.0;
    double xy = 0.0;

    double q;

    // Initialize result array
    largest_q_result[0] = -DBL_MAX;
    largest_q_result[1] = -DBL_MAX;
    largest_q_result[2] = -DBL_MAX;

    // Index from the end of the 'left' cluster (X)
    int end_x = min_cluster_size - 1;
    // Index from the end of the 'right' cluster (Y)
    int end_y = end_x + min_cluster_size;

    // ###########################
    // 1. Calculate Q value for X and Y with each having min_cluster_size elements.
    // We need to perfom this operation at the beginning to prepare the cache.
    // ###########################
    for (int row = 0; row < end_x; row++) {
        x += sum_fixed_row(distances, length, row, row + 1, end_x + 1);
    }
    for (int row = end_x + 1; row < end_y; row++){
        y += sum_fixed_row(distances, length, row, row + 1, end_y + 1);
    }
    for (int row = 0; row < end_x + 1; row++) {
        xy += sum_fixed_row(distances, length, row, end_x + 1, end_y + 1);
    }

    x_size = end_x + 1;
    y_size = (end_y + 1) - x_size;
    q = ((2.0 / (x_size * y_size)) * xy) - ((2.0 / (x_size * (x_size - 1))) * x) - ((2.0 / (y_size * (y_size - 1))) * y);
    q *= (x_size * y_size) / (x_size + y_size);
    if (q > largest_q_result[1]) {
        largest_q_result[0] = 1;
        largest_q_result[1] = q;
        largest_q_result[2] = end_x + 1;
    }

    double y_cache[length];
    y_cache[end_y] = y;
    double xy_cache[length];
    xy_cache[end_y] = xy;

    // ###########################
    // 2. Keep number of elements of the left cluster (X) constant and increase the number of elements of the right cluster (Y)
    // - x-value: Does not change as left cluster (X) stays constant.
    // - y-value: Incrementally compute the y-value by using the old y-value and adding the difference of the newly added data point.
    // - xy:value: See y-value.
    // ###########################
    end_y++;
    for (;end_y < length; end_y++) {
        y_cache[end_y] = y_cache[end_y - 1] + sum_fixed_column(distances, length, end_x + 1, end_y, end_y);
        xy_cache[end_y] = xy_cache[end_y - 1] + sum_fixed_column(distances, length, 0, end_x + 1, end_y);

        y = y_cache[end_y];
        xy = xy_cache[end_y];
        x_size = end_x + 1;
        y_size = (end_y + 1) - x_size;
        q = ((2.0 / (x_size * y_size)) * xy) - ((2.0 / (x_size * (x_size - 1))) * x) - ((2.0 / (y_size * (y_size - 1))) * y);
        q *= (x_size * y_size) / (x_size + y_size);
        if (q > largest_q_result[1]) {
            largest_q_result[0] = 1;
            largest_q_result[1] = q;
            largest_q_result[2] = end_x + 1;
        }
    }

    // ###########################
    // 3. Increase left side and perform step 2 now for each increase of the left side.
    // x-value: Incrementally compute the x-value by using the old x-value and adding the difference of the newly added data point.
    // y -value: See x-value.
    // xy-value: See x-value.
    // ###########################
    end_x++;
    for (;; end_x++) {
        end_y = end_x + min_cluster_size;
        if (end_y >= length) {
            break;
        }

        double add_x = sum_fixed_column(distances, length, 0, end_x, end_x);
        x += add_x;
        for (; end_y < length; end_y++) {
            double add_y = sum_fixed_row(distances, length, end_x, end_x + 1, end_y + 1);
            y_cache[end_y] -= add_y;
            xy_cache[end_y] += (add_y - add_x);

            y = y_cache[end_y];
            xy = xy_cache[end_y];
            x_size = end_x + 1;
            y_size = (end_y + 1) - x_size;
            q = ((2.0 / (x_size * y_size)) * xy) - ((2.0 / (x_size * (x_size - 1))) * x) - ((2.0 / (y_size * (y_size - 1))) * y);
            q *= (x_size * y_size) / (x_size + y_size);
            if (q > largest_q_result[1]) {
                largest_q_result[0] = 1;
                largest_q_result[1] = q;
                largest_q_result[2] = end_x + 1;
            }
        }
    }
    return true;
}

/**
 * Calculate the sum of terms in a NxN distance matrix within
 * a fixed row [row, row) x [column_start, column_end).
 *
 * @param distances The NxN distance matrix.
 * @param length The length of one dimension of the distance matrix, i.e. the integer N.
 * @param row Index of the row.
 * @param column_start Index of the column where the square begins (inclusive).
 * @param column_end Index of the column where the square ends (exclusive).
 */
double sum_fixed_row(double * distances, int length, int row, int column_start, int column_end) {
    double sum = 0.0;
    for(int column = column_start; column < column_end; column++) {
        sum = sum + distances[row * length + column];
    }
    return sum;
}

/**
 * Calculate the sum of terms in a NxN distance matrix within
 * a fixed column [row_start, row_end) x [column, column).
 *
 * @param distances The NxN distance matrix.
 * @param length The length of one dimension of the distance matrix, i.e. the integer N.
 * @param row_start Index of the row where the square begins (inclusive).
 * @param row_end Index of the row where the square ends (exclusive).
 * @param column Index of the column.
 */double sum_fixed_column(double * distances, int length, int row_start, int row_end, int column) {
    double sum = 0.0;
    for(int row = row_start; row < row_end; row++) {
        sum = sum + distances[row * length + column];
    }
    return sum;
}


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
