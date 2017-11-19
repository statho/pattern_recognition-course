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
figure;
imagesc(reshape(reshaped_train_data(2:257,131), 16, 16)');


% find mean and variance of features for the digits (0-9)
% and store values in means and vars

means = [];
vars = [];

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

    means(:,i+1) = mean(pixels, 2);
    vars(:,i+1) = var(pixels, 0, 2);
   
end


% Step 2 - calculate the mean value of pixel(10,10) features for digit "0"
means0 = reshape(means(:,1), 16, 16);
mean0 = means0(10,10)

% Step 3 - calculate the variance of pixel(10,10) features for digit "0"
vars0 = reshape(vars(:,1), 16, 16);
var0 = vars0(10,10)

% Step 5 - show digit "0" using mean values 
figure;
imagesc(reshape(means(:,1), 16, 16)');

% Step 6 - show digit "0" using variances
figure;
imagesc(reshape(vars(:,1)', 16, 16)');

% Step 7 - show all digits using the mean values of their features
for i = 2:10
    figure;
    imagesc(reshape(means(:,i), 16, 16)');
end

% Step 9 - classify all the test digits

actual = reshaped_test_data(1,:);
estimated = [];
for i = 1:size(reshaped_test_data, 2)
    estimated(i) = euclidean_classifier(reshaped_test_data(2:257, i), means);
end

% Step 8 - show the digit's in position 101 actual and estimated label
display(actual(101));
display(estimated(101));


% calculate total Recall

correct = 0;
for i = 1:length(actual)
    if actual(i) == estimated(i)
        correct = correct + 1;
    end
end

total_recall = correct / length(actual);
display(total_recall);
