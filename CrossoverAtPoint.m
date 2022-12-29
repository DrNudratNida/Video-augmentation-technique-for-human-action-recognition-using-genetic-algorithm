function [result1 result2] = CrossoverAtPoint(sequence1, sequence2, point)
    result1 = [sequence1(1:point-1), sequence2(point:end)];
    result2 = [sequence2(1:point-1), sequence1(point:end)];
end