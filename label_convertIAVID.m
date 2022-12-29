function [trainLabelNew, testLabelNew] = label_convertIAVID(trainLabel, testLabel, type)

if nargin < 3
    type = '2';
end

classes    = [1 2 3 4 5 6 7 8];
nClasses   = 8;
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

