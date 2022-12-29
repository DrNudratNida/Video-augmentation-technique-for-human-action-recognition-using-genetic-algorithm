function [Mcross]=MatrixCrossOver(matrixA,point)
[r c]=size(matrixA)
Mcross=zeros(r, c);
%point=3;
    for j=1:2:r-1
    [result1 result2] = CrossoverAtPoint(matrixA(j,:), matrixA(j+1,:), point);
    p=j;
    Mcross(p,:)=result1 ;
    Mcross(p+1,:)=result2;
    %j=j+2;
    end
end
