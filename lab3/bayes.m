function [percent] = bayes(testArray, means, variances, a_priori)

%variances = ones(2,size(testArray,2)-1);
prob_array=(-1)*ones(1,2);
cnt = 0;
for number=1:size(testArray,1)
    for i=1:2
        prob_array(i)=mvnpdf(testArray(number,2:size(testArray,2)), means(i,:), variances(i,:)) ;   % P(x|Ci) OR with VARIANCE = [] !!
    end
    overall_prob(:) = prob_array(:).*(a_priori(:));       % P(Ci|x) = P(x|Ci)*P(Ci)
    if ( (overall_prob(1)>overall_prob(2) && testArray(number,1) == -1) || (overall_prob(1)<overall_prob(2) && testArray(number,1) == +1)) 
       cnt = cnt +1 ;
    end
end
percent = 100*(cnt/size(testArray,1));        %Bayes classifier percentage