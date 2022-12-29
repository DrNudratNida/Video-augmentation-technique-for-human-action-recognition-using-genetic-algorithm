function [result1] = Crossover2Point(sequence1, sequence2, point1,point2)
    result1 = [sequence1(1:point1-1), sequence2(point1:point2-1),sequence1(point2:end)];
    
end