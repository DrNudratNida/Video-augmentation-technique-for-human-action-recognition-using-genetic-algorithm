function [GATable, OptFeaturesTrain,OptFeaturesTest,location,maxAcc]=EvolutionaryEDTMD(NumGeneration,FitVal,traindata,testdata,trainlabel,testlabel)
i=0;
gpuDevice(1) 
while (FitVal~=100 && i<=NumGeneration)%100
   % traindata=Data1.trainingFeatures;%(traindata);%
%testdata=Data1.testFeatures;
matrixA=traindata';
matrixB=testdata';
[p1 c]=size(matrixA);
[p2 cc]=size(matrixB);

i=i+1
phalf=(floor(p1/2)-1);
%matrixA=randi([1 100],[4000 4000]);
 pt1=randi([1,2000]);
 pt2=randi([2000,4000]);
% pt1=randi([1,phalf]);
% pt2=randi([phalf,p1]);
% pt1=randi([1,500]);
% pt2=randi([500,999]);
%pt1=randi([1,250]);
%pt2=randi([250,500]);
[poptrain]=MatrixTwoPointCrossOver(matrixA,pt1,pt2);
[poptest]=MatrixTwoPointCrossOver(matrixB,pt1,pt2);
% size(poptrain);
% size(poptest);

[FitVal, traindataUpdated, testdataUpdated,label_actual_test,label_desired_test]=FitFunc_ELM(poptrain, poptest,trainlabel,testlabel);
fprintf('      Generation      |    Fitness Value   \n');
fprintf('-------------------------------------------\n');
DataUpdatedGA(i).Generation=i;
DataUpdatedGA(i).FitVal=FitVal;
DataUpdatedGA(i).Trainpop=traindataUpdated;
DataUpdatedGA(i).Testpop=testdataUpdated;
DataUpdatedGA(i).ActualLabel=label_actual_test;
DataUpdatedGA(i).PredictedLabel=label_desired_test;
 
end
GATable=struct2table(DataUpdatedGA);
[maxAcc ind]=max((GATable.FitVal));
%location=GATable(strcmp(cell2mat(GATable.FitVal),maxAcc),:);
r=find((GATable.FitVal)==maxAcc);
location=GATable(ind,:);
OptFeaturesTrain=cell2mat(GATable.Trainpop(ind));
OptFeaturesTest=cell2mat(GATable.Testpop(ind));