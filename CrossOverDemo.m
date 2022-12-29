clc
close all
clear all
%seq1 = [1 2 3 7 9 10 5 4 6 8]

%seq2 = [4 3 9 2 1 7 6 10 8 5]
%[result1 result2]=CrossoverAtPoint(seq1, seq2, 5)


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
load('AlexNetFeaturesMelanoma_fc6.mat');
TrainFeatures=trainingFeatures;
%TrainLabels=double(trainingLabels');
TestFeatures=(testFeatures);
%TestLabels=double(testLabels');
%[trainLabelNew, testLabelNew] = label_convert(TrainLabels, TestLabels,2);
%net = trainSoftmaxLayer(TrainFeatures,trainLabelNew);
%Predict=net(TestFeatures);

%actual=testLabelNew;
%plotconfusion(actual,Predict);
%numcorrect = sum(actual==Predict);
%accuracy3 = numcorrect/379;
matrixA=TrainFeatures';
%matrixA=randi([1 100],[4000 4000]);
pt1=randi([1,2000]);
pt2=randi([2000,4000]);
[TwoPtcross]=MatrixTwoPointCrossOver(matrixA,pt1,pt2)
[OnePtcross]=MatrixCrossOver(matrixA,pt1)
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

function [Mcross]=MatrixTwoPointCrossOver(matrixA,point1,point2)
[r c]=size(matrixA)
Mcross=zeros(r, c);
%point=3;
    for j=1:2:r-1
    [result1 result2] = CrossoverAtTwoPoint(matrixA(j,:), matrixA(j+1,:), point1,point2);
    p=j;
    Mcross(p,:)=result1 ;
    Mcross(p+1,:)=result2;
    %j=j+2;
    end
end

function [result1 result2] = CrossoverAtPoint(sequence1, sequence2, point)
    result1 = [sequence1(1:point-1), sequence2(point:end)];
    result2 = [sequence2(1:point-1), sequence1(point:end)];
end

function [result1 result2] = CrossoverAtTwoPoint(sequence1, sequence2, point1,point2)
    size(sequence1)
    size(sequence2)
    size(sequence1(1:point1-1))
    
    result1 = [sequence1(1:point1-1), sequence2(point1:point2-1),sequence1(point2:end)];
    result2 = [sequence2(1:point1-1), sequence2(point1:point2-1),sequence2(point2:end)];
end
