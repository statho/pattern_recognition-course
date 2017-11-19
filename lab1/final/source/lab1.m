%% Pattern Recognition - Lab 1

clc; close all; clear all;

% read train file and reshape it into digit vectors
train_file = fopen('train.txt', 'r');
train_data = fscanf(train_file, '%f');
fclose(train_file);

reshaped_train_data = reshape(train_data, 257, []);

% read tets file and reshape it into digit vectors
test_file = fopen('test.txt', 'r');
test_data = fscanf(test_file, '%f');
fclose(test_file);

reshaped_test_data = reshape(test_data, 257, []);

% Step1 - print digit 131
%figure;
%imagesc(reshape(reshaped_train_data(2:257,131), 16, 16)');


% find mean and variance of features for the digits (0-9)
% and store values in means and vars

means = [];
vars = [];
%{
for i = 0:9
    
    pixels = [];
    j = 1;
    for k = 1:size(reshaped_test_data,2)
        number = reshaped_train_data(1,k);
        if number == i
            % pixels has the features of one digit at a time
            pixels(:,j) = reshaped_train_data(2:257,k);
            j = j + 1;
        end
    end
    m = mean(pixels, 2);
    s = var(pixels, 0, 2);
    means(:,i+1) = m(:);
    vars(:,i+1) = s(:);
   
end

%}

for i = 0:9
    [means_1, vars_1] = find_mean_var(reshaped_train_data, i);
    means(:,i+1) = means_1(:);
    vars(:,i+1) = vars_1(:);
end

% Step 2 - calculate the mean value of pixel(10,10) features for digit "0"
means0 = reshape(means(:,1), 16, 16);
mean0 = means0(10,10)

% Step 3 - calculate the variance of pixel(10,10) features for digit "0"
vars0 = reshape(vars(:,1), 16, 16);
var0 = vars0(10,10)

% Step 5 - show digit "0" using mean values 
%figure;
%imagesc(reshape(means(:,1), 16, 16)');

% Step 6 - show digit "0" using variances
%figure;
%imagesc(reshape(vars(:,1)', 16, 16)');

% Step 7 - show all digits using the mean values of their features
%for i = 2:10
 %   figure;
 %   imagesc(reshape(means(:,i), 16, 16)');
%end

% Step 9 - classify all the test digits

actual = reshaped_test_data(1,:);
estimated = [];
for i = 1:size(reshaped_test_data, 2)
    estimated(i) = euclidean_classifier(reshaped_test_data(2:257, i), means);
end

% Step 8 - show the digit's in position 101 actual and estimated label
%display(actual(101));
%display(estimated(101));


% calculate total Recall

correct = 0;
for i = 1:length(actual)
    if actual(i) == estimated(i)
        correct = correct + 1;
    end
end

total_recall = correct / length(actual);
display(total_recall);

% Step 10 - calculate a-priori propabilities
appearances = reshaped_test_data(1,:);
digit_appearance = zeros(10,1);
for i = 1:10
    digit_appearance(i) = sum(appearances == (i-1));
end
a_priori = digit_appearance/length(appearances);



% Step 11 - implement Bayesian classifier
for j = 1:10
    current(:, 1) = vars(:, j);
    current(current == 0) = 1;
    %sigma(:,:,j) = diag(current(:, 1));
    %mysigma = sigma(:,:,j);
    likelihood(:, j) = mvnpdf(reshaped_test_data(2:257, :)', means(:, j)', diag(current(:,1))) * a_priori(j);
end


[V, row] = max(likelihood, [], 2);
meter = 0;
for i = 1:length(actual)
    if actual(i) ==  row(i) -1
        meter = meter +1;
    end
end
total_bayes = meter / length(actual);
display(total_bayes);

%Step 12-Bayesian classifier with variance equal to 1.
for j = 1:10
    no_var_likelihood(:, j) = mvnpdf(reshaped_test_data(2:257, :)', means(:, j)', eye(256)) * a_priori(j);
end


[V, row] = max(no_var_likelihood, [], 2);
meter = 0;
for i = 1:length(actual)
    if actual(i) ==  row(i) -1
        meter = meter +1;
    end
end
total_bayes_new = meter / length(actual);
display(total_bayes_new);


%Step 13
train_sample = reshaped_train_data(2:257, 1:1000);
for j=1:100
    test_sample = reshaped_test_data(2:257, j);
    for i = 1:1000
        neighbors(i) = norm(test_sample - train_sample(:, i), 2);
    end
    [A, nearest_one(j,:)] = sort(neighbors);
end

meter = 0;
for i = 1:100
    number = reshaped_train_data(1,nearest_one(i, 1));
    if actual(i) == number
        meter = meter + 1;
    end
end

totan_1nn_100 = meter / 100;
display(totan_1nn_100);

%Step 14 (a), (b)
for j=1:size(reshaped_test_data, 2)
    
    for i = 1:size(reshaped_train_data, 2)
        neighbors_all(i) = norm(reshaped_test_data(2:257, j) - reshaped_train_data(2:257, i), 2);
    end
    [A, nearest_one_all(j,:)] = sort(neighbors_all);
end

meter = 0;
for i = 1:size(reshaped_test_data, 2)
    number = reshaped_train_data(1,nearest_one_all(i, 1));
    if actual(i) == number
        meter = meter + 1;
    end
end
totan_1nn_all = meter / size(reshaped_test_data, 2);
display(totan_1nn_all);

%Step 14 (c)
result_array = knn_algo(nearest_one_all, 3);
meter = 0;
for i = 1:size(reshaped_test_data, 2)
    number = reshaped_train_data(1,result_array(i));
    if actual(i) == number
        meter = meter + 1;
    end
end
total_3nn_all = meter / size(reshaped_test_data, 2);
display(total_3nn_all);

% k = 5
result_array = knn_algo(nearest_one_all, 5);
total_5nn_all = knn_total(result_array, reshaped_train_data, actual);
display(total_5nn_all);

%k = 7
result_array = knn_algo(nearest_one_all, 7);
total_7nn_all = knn_total(result_array, reshaped_train_data, actual);
display(total_7nn_all);

% k = 9
result_array = knn_algo(nearest_one_all, 9);
total_9nn_all = knn_total(result_array, reshaped_train_data, actual);
display(total_9nn_all);
%k = 11
result_array = knn_algo(nearest_one_all, 11);
total_11nn_all = knn_total(result_array, reshaped_train_data, actual);
display(total_11nn_all);
