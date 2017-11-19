function [ m, s ] = find_mean_var( data, digit)
pixels = [];
j = 1;
for i = 1 : size(data,2)
    number = data(1,i);
    features = data(2:257,i);
    if number == digit
        pixels(:,j) = features;
        j = j + 1;
    end
end

m = mean(pixels, 2);
s = var(pixels, 0, 2);
end
