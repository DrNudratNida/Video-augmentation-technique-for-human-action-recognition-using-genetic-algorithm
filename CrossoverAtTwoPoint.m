function [result1 result2] = CrossoverAtTwoPoint(sequence1, sequence2, point1,point2)
%     size(sequence1)
%     size(sequence2)
%     size(sequence1(1:point1-1))
    
    result1 = [sequence1(1:point1-1), sequence2(point1:point2-1),sequence1(point2:end)];
    result2 = [sequence2(1:point1-1), sequence2(point1:point2-1),sequence2(point2:end)];
end