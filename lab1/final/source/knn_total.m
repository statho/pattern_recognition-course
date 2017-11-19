function [total] = knn_total(result_array, reshaped_train_data, actual)

meter = 0;
for i = 1:size(actual, 2)
    number = reshaped_train_data(1,result_array(i));
    if actual(i) == number
        meter = meter + 1;
    end
end
total = meter / size(actual, 2);
