function [result_array] = knn_algo(nearest, k)


loop = nearest(:, 1:k);


for i = 1:size(loop, 1)
    result = loop(i, 1);
    counter = 1;
    for j = 2:k
        cand = loop(i, j);
        if result == cand
            counter = counter + 1;
        else
            if counter == 1
                result = cand;
            else
                counter = counter - 1;
            end
        end
    end
    result_array(i) = result;
end
