function [Mcross]=MatrixTwoPointCrossOver(matrixA,point1,point2)
[r c]=size(matrixA)
Mcross=zeros(r, c);
%point=3;
    for j=1:1:r-1
        
    [result1 result2] = CrossoverAtTwoPoint(matrixA(j,:), matrixA(j+1,:), point1,point2);
    p=j;
    Mcross(p,:)=result1 ;
   Mcross(p+1,:)=result2;
    %j=j+2;
    end
end