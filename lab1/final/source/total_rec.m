function [ total_recall] = total_rec( actual, estimated )

correct = 0;
for i = 1:length(actual)
    if actual(i) == estimated(i)
        correct = correct + 1;
    end
end
total_recall = correct / length(actual);
