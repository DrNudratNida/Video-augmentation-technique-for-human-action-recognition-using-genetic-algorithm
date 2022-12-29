%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%       Main file to perform leave one actor out human action recognition
%       usng MuHAVi Uncut dataset
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

s='F:\PhD\experiments\GeneticAlgoImp\Alex_Muhavi\';%Specify the path of MHIs
dirData = dir(s); 
%# Get the selected file data
fileNames = {dirData.name}; %# Create a cell array of file names
fileNames=fileNames(3:end);
%%
%
tic
for i=1:numel(fileNames)
 r=regexp(fileNames(i),'_','split');%regexp(cellArr, '[_.]', 'split')
 Label{i}=r{1,1}{1,2}; 
 person{i}=r{1,1}{1,3};
end
tic
for j=1:7
clear r1;clear r2;clear testFiles; clear trainingSets;clear validationSets; clear TrainLabel; clear TestLabel;clear trainFiles;
%CameraName=strcat('Camera', int2str(j));
PersonName=strcat('Person', int2str(j));
index = find(strcmp(person, PersonName));
%Camera_Index= find(strcmp(Cm_name, CameraName));
%TestData=find(Cm_name,1);
%   idx(i) = find(not(cellfun(@isempty, strfind(r{4}, 'Camera1'))));
clear trainFiles;clear testFile;
testIndex=index;
%trainIndex=find(strcmp(strcat('Sample_',sequence{j},'tif.png'), SequenceName)==0);
trainIndex=find(strcmp(person, PersonName)==0);
trainFiles=fileNames(trainIndex);
testFiles=fileNames(testIndex);
for i=1:numel(trainIndex)
r1=regexp(trainFiles(i),'_','split');%regexp(cellArr, '[_.]', 'split')
 TrainLabel{i}=r1{1,1}{1,2};
%TrainLabel(1,i)=(TrainLabel);
 %{i}=imread(strcat(s,char(trainFiles(i))));
 label=char(TrainLabel{i});
 TrainLabel{i}=(classAssignment(label));
end
TrainLabel=cell2mat(TrainLabel);
clear TestLabel;
for i=1:numel(testIndex)
r2=regexp(testFiles(i),'_','split');%regexp(cellArr, '[_.]', 'split')
 TestLabel{i}=r2{1,1}{1,2};
 labelTest=char(TestLabel{i});
 TestLabel{i}=(classAssignment(labelTest));
% TestData{i}=imread(strcat(s,char(testFiles(i))));
end
TestLabel=cell2mat(TestLabel);
% vv = gpuDevice(1);
% reset(vv);
gpuDevice(1)
net=alexnet;
%gpuDevice(1) 
layer ='fc6';%'%loss3-classifier';% 'fc1000';%'pool5-7x7_s1';
a=numel(trainIndex);
b=numel(testIndex);
%trainingFeatures=zeros(a,1920);
%testFeatures=zeros(b,1920);
for i=1:numel(trainIndex)
    img=imread(fullfile('Alex_Muhavi',trainFiles{i}));%dirData(i).name
ff =activations(net,img,layer);
ff=ff(:)';
trainingFeatures(i,:)=ff(1,1:4096);%(1,1,:);%(1,1:4096);%(1,1,:);
end
for i=1:numel(testIndex)
    img1=imread(fullfile('Alex_Muhavi',testFiles{i}));
    kk= activations(net,img1,layer);
   kk=kk(:)';
testFeatures(i,:) =kk(1,1:4096);%(1,1,:);%(1,1:4096);%(1,1,:);
end
%trainingFeatures=trainingFeatures;
%testFeatures=testFeatures;
%%
% Extract the class labels from the training and test data.
trainlabel = (TrainLabel)';
testlabel = (TestLabel)';
traindata=trainingFeatures';%permute(trainingFeatures,[3 2 1]);%(traindata);%
testdata=testFeatures';%permute(testFeatures,[3 2 1]); %(testdata); %
nn.label = trainlabel;

[trainlabel, testlabel] = label_convertMU(trainlabel,testlabel,'2');
%-----------Setting----------------------------------------------------
method            = {'ELM','RELM'};

nn.hiddensize     = 129060;
nn.activefunction = 't';
lamda   = 30e-2;
tol     = 5e-2;
nn.inputsize      = size(traindata,1);
nn.method         = method{1};
nn.type           = 'classification';
%-----------Initialization----------

nn                = elm_initialization(nn);
fprintf('      method      |    Optimal C    |  Training Acc.  |    Testing Acc.   |   Training Time \n');
fprintf('--------------------------------------------------------------------------------------------\n');

nn.method         = method{1};
%traindata=double(yael_fvecs_normalize(traindata));
%testdata=double(yael_fvecs_normalize(testdata));
[nn, acc_train,label_actual,label_desired]   = elm_train(traindata, trainlabel, nn);
[nn1, acc_test,label_actual_test,label_desired_test]    = elm_test(testdata, testlabel, nn);
%predicted=predict(categoryClassifier, validationSets);
actualLabels{j,:}=label_actual_test';
predictedLabels{j,:}=(label_desired_test)';


%confMatrix2= evaluate(categoryClassifier, validationSets);
%colormap(confMatrix);

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                             Saving features
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
Train_Features=traindata;
Test_Features=testdata;
Train_Label=trainlabel;
Test_Label=testlabel;
s=strcat('E:\Features\MUHAVI_densenet201_bn_LOAO_Actor_',int2str(j),'.mat');
save(s,'predictedLabels','actualLabels','Train_Features','Train_Label','Test_Features','Test_Label');
numcorrect = sum(label_actual_test==label_desired_test);
accuracy = (numcorrect/length(testlabel));

accuracy = mean(numcorrect/length(testlabel));
total=length(testlabel);
FitVal1=accuracy;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                   Evolutionary deep features
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
NumGeneration=100;
[GATable, OptFeaturesTrain,OptFeaturesTest,location,maxFitness]=EvolutionaryEDTMD(NumGeneration,FitVal1,traindata,testdata,trainlabel,testlabel);%figure
[nn, GAacc_train,GAlabel_actual,GAlabel_desired]   = elm_train(OptFeaturesTrain, trainlabel, nn);
[nn1, GAacc_test,GAlabel_actual_test,GAlabel_desired_test]    = elm_test(OptFeaturesTest, testlabel, nn);
GAactualLabels{j,:}=GAlabel_actual_test';
GApredictedLabels{j,:}=(GAlabel_desired_test)';
g=strcat('E:\Features\GACNN_MUHAVI_densenet201_bn_LOAO_Actor_',int2str(j),'.mat');

save(g,'OptFeaturesTrain','OptFeaturesTest','location','maxFitness','GAactualLabels','GApredictedLabels');

%%
clear Train_Features; clear Train_Label; clear Test_Features; clear Test_Label;
clear testFiles; clear trainingSets;clear validationSets; clear TrainLabel; clear TestLabel;clear trainFiles;
%I=confPlot(confMatrix2 ,order);
%confusionMatrix=strcat('F:\PhD\experiments\CodeCV\DeepBoF\MelanomaCode\results\ConfusionMatrix_400_',s,'.png')
%imwrite('confusionMatrix',I);
clear r;
clear r1;
clear r2;
end
GATime=toc
predicted=[predictedLabels{1,1}',predictedLabels{2,1}',predictedLabels{3,1}',predictedLabels{4,1}',predictedLabels{5,1}',predictedLabels{6,1}',predictedLabels{7,1}'];
order={'ClimbLadder','CrawlOnKnees','DrawGraffiti','DrunkWalk','JumpOverFence','JumpOverGap','Kick','LookInCar','PickupThrowObject','PullHeavyObject','Punch','RunStop','ShotGunCollapse','SmashObject','WalkFall','WalkTurnBack','WaveArms'};
%predicted=order(predicted);
actual=[actualLabels{1,1}',actualLabels{2,1}',actualLabels{3,1}',actualLabels{4,1}',actualLabels{5,1}',actualLabels{6,1}',actualLabels{7,1}'];
%actual=order(actual);
cM=confusionmat((actual),(predicted));
figure
confPlot(cM,order);
%%
plotroc(actual,predicted)
% s=confusionmat(a,b)
for i =1:size(cM,1)
 
     precision(i)=cM(i,i)/sum(cM(:,i));
 end
 for i =1:size(cM,1)
% 
     Recall(i)=cM(i,i)/sum(cM(i,:));
 end
% 
 Precision=sum(precision)/size(cM,1);
 Recall=mean(Recall);

 F_score=2*Recall*Precision/(Precision+Recall);
 %F_score=2*1/((1/Precision)+(1/Recall));
 error=1-F_score;
numcorrectN = sum((actual)==(predicted));
accuracyN = (numcorrectN/length(actual));

GApredicted=[GApredictedLabels{1,1}',GApredictedLabels{2,1}',GApredictedLabels{3,1}',GApredictedLabels{4,1}',GApredictedLabels{5,1}',GApredictedLabels{6,1}',GApredictedLabels{7,1}'];
GAactual=[GAactualLabels{1,1}',GAactualLabels{2,1}',GAactualLabels{3,1}',GAactualLabels{4,1}',GAactualLabels{5,1}',GAactualLabels{6,1}',GAactualLabels{7,1}'];
%actual=order(actual);
GAcM=confusionmat((GAactual),(GApredicted));
figure
confPlot(GAcM,order);
numcorrectGA = sum((GAactual)==(GApredicted));
accuracyGA = (numcorrectGA/length(GAactual));


 plotroc(GAactual,GApredicted)
% s=confusionmat(a,b)
for i =1:size(GAcM,1)
 
     GAprecision(i)=GAcM(i,i)/sum(GAcM(:,i));
 end
 for i =1:size(GAcM,1)
% 
     GARecall(i)=GAcM(i,i)/sum(GAcM(i,:));
 end
 GARecall=mean(GARecall);
GAPrecision=sum(GAprecision)/size(GAcM,1);
% 
% %%% F-score
% f=2*1*0.967/1.967;
 GAF_score=2*GARecall*GAPrecision/(GAPrecision+GARecall);
 %GAF_score=2*1/((1/GAPrecision)+(1/GARecall));
 GAerror=1-GAF_score;
 FilePath='E:\Features\Results\LOAO_MUHAVI_densenet201_bn.mat';
save(FilePath,'accuracyGA','GARecall','GAPrecision','GAF_score','accuracyN','Recall','Precision','actual','F_score','predicted','GApredicted','GAactual');
