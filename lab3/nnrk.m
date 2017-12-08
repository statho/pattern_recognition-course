function [percent] = nnrk (trainArray, testArray, k)


    %k=5;   %how many neighbours 
    nnrk_count = 0;
    nnrk = zeros(1,size(trainArray,1));

    for test=1:size(testArray,1)
        nearest_class_count = zeros(1,2);
        for train=1:size(trainArray,1)
            nnrk(train) = sqrt(sum((trainArray(train,2:size(trainArray,2)) - testArray(test,2:size(testArray,2))) .^ 2));
        end
        nnrk2 = sort(nnrk); %se poia thesh htan prin, auto to trainArray thelw
        for neigh = 1:k
            pos1 = trainArray( (nnrk == nnrk2(neigh) ), 1);
            if (pos1 == -1)
                pos = 1;
            else
                pos = 2;
            end
            nearest_class_count(pos) = nearest_class_count(pos) + 1;
        end
        if( ( nearest_class_count(1) > nearest_class_count(2) && testArray(test,1) == -1 ) || ( nearest_class_count(1) < nearest_class_count(2) && testArray(test,1) == 1) )
%         if( find(nearest_class_count == max(nearest_class_count))-1 == testArray(test,1) )
            nnrk_count = nnrk_count +1;
        end
    end
    percent = 100*(nnrk_count/size(testArray,1));
    
