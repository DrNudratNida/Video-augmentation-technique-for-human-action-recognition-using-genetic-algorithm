function [trainLabelNew, testLabelNew] = label_convertMU(trainLabel, testLabel, type)

if nargin < 3
    type = '2';
end

classes    = [1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38];
nClasses   = 38;
nTrainData = numel(trainLabel);
nTestData  = numel(testLabel);

trainLabelNew = -ones(nClasses,nTrainData,'single');
testLabelNew  = -ones(nClasses,nTestData,'single');

for i = 1 : nClasses
    trainLabelNew(i,trainLabel==classes(i)) = 1;
    testLabelNew(i,testLabel==classes(i))   = 1;
end

if ~strcmp(type,'2')
    trainLabelNew = (trainLabelNew+1)/2;
    testLabelNew  = (testLabelNew+1)/2;
end


end

