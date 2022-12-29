s='F:\PhD\experiments\GAN\GeneticAlgoImp\Alex_Ixmas2\';
dirData = dir(s); 
%# Get the selected file data
fileNames = {dirData.name}; %# Create a cell array of file names
fileNames=fileNames(3:end);

for i=1:numel(fileNames)
 r=regexp(fileNames(i),'_','split');%regexp(cellArr, '[_.]', 'split')
 Label{i}=r{1,1}{1,4}; 
 expression='.avi.tif';
 p=r{1,1}{1,5};
 a=regexp(p,expression,'split');
 %Cm_name{i}=a{1,1};
 person{i}=r{1,1}{1,2};
 %sequence{i}=r{1,1}{1,3};
end
tic
n=0;
for j=10:10
    clear r1;clear r2;clear testFiles; clear trainingSets;clear validationSets; clear TrainLabel; clear TestLabel;clear trainFiles;clear trainIndex;clear testIndex;    
    PersonName={'alba','amel','andreas','chiara','clare','daniel','florian','julien','hedlena','nicolas','pao','srikumar'};
%index = find(strcmp(person, PersonName));
Index=find(strncmp(person, PersonName{1,j},3));
trainIndex=find(strncmp(person, PersonName{1,j},3)==0);
n=n+1;
%Camera_Index= find(strcmp(Cm_name, CameraName));
%TestData=find(Cm_name,1);
%   idx(i) = find(not(cellfun(@isempty, strfind(r{4}, 'Camera1'))));
clear trainFiles;clear testFile;
testIndex=Index;
%trainIndex=find(strcmp(strcat('Sample_',sequence{j},'tif.png'), SequenceName)==0);

trainFiles=fileNames(trainIndex);
testFiles=fileNames(testIndex);
for i=1:numel(trainIndex)
r1=regexp(trainFiles(i),'_','split');%regexp(cellArr, '[_.]', 'split')
 TrainLabel{i}=r1{1,1}{1,4};
%TrainLabel(1,i)=(TrainLabel);
 %{i}=imread(strcat(s,char(trainFiles(i))));
 label=char(TrainLabel{i});
 TrainLabel{i}=(classAssignmentIxmas(label));
end
TrainLabel=cell2mat(TrainLabel);
clear TestLabel;
for i=1:numel(testIndex)
r2=regexp(testFiles(i),'_','split');%regexp(cellArr, '[_.]', 'split')
 TestLabel{i}=r2{1,1}{1,4};
 labelTest=char(TestLabel{i});
 TestLabel{i}=(classAssignmentIxmas(labelTest));
% TestData{i}=imread(strcat(s,char(testFiles(i))));
end
TestLabel=cell2mat(TestLabel);
net=densenet201;
layer = 'bn';
aa=numel(trainIndex);
bb=numel(testIndex);
%trainingFeatures=zeros(aa,1000);
%testFeatures=zeros(bb,1000);
for i=1:numel(trainIndex)
    img=imread(fullfile('Alex_Ixmas2',trainFiles{i}));%dirData(i).namen
    f=activations(net,img,layer);
    f=f(:);
    f=f(1:10000,1);
trainingFeatures(i,:)=f; 
clear f;
end
for i=1:numel(testIndex)
    img=imread(fullfile('Alex_Ixmas2',testFiles{i}));
ff= activations(net,img,layer);
ff=ff(:);
ff=ff(1:10000,1);
testFeatures(i,:) =ff;
clear ff;
end
trainingFeatures=trainingFeatures';
testFeatures=testFeatures';
%%
% Extract the class labels from the training and test data.
trainlabel = (TrainLabel)';
testlabel = (TestLabel)';
traindata=trainingFeatures;%(traindata);%
testdata=testFeatures; %(testdata); %
%traindata=(traindata);%
%testdata=(testdata); %

nn.label = trainlabel;

% [traindata,PS] = mapminmax(traindata,-1,1);%
% testdata = mapminmax('apply',testdata,PS);
% 
% traindataPca=(pca(traindata'))';
% traindata=traindataPca*(traindata);
% testdata=traindataPca*testdata;
[trainlabel, testlabel] = label_convertiXmas(trainlabel,testlabel,'2');
%-----------Setting----------------------------------------------------
method            = {'ELM','RELM'};
%traindata  =  traindata./( repmat(sqrt(sum(traindata.*traindata)), [size(traindata,1),1]) );
%testdata   =  testdata./(repmat(sqrt(sum(testdata.*testdata)), [size(testdata,1),1]) );

 %[traindata,PS] = mapminmax(traindata,-1,1);%
 %testdata = mapminmax('apply',testdata,PS);

% [traindata,PS] = mapstd(traindata);%
% testdata = mapstd('apply',testdata,PS);
nn.hiddensize     = 129061;
%nn.hiddensize     = 40,000;
nn.activefunction = 's';
lamda   = 30e-5;
tol     = 50e-5;
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
tic
[nn1, acc_test,label_actual_test,label_desired_test]    = elm_test(testdata, testlabel, nn);
toc
%predicted=predict(categoryClassifier, validationSets);
actualLabels{j,:}=label_actual_test';
predictedLabels{j,:}=(label_desired_test)';

Train_Features=traindata;
Test_Features=testdata;
Train_Label=trainlabel;
Test_Label=testlabel;
s=strcat('E:\Features\LOAO_IXMAS\IXMAS_densenet201BN_LOAO','.mat');
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
g=strcat('E:\Features\LOAO_IXMAS\GACNN_IXMAS_densenet201BN_LOAO','.mat');

save(g,'OptFeaturesTrain','OptFeaturesTest','location','maxFitness','GAactualLabels','GApredictedLabels');
                
%%
clear Train_Features; clear Train_Label; clear Test_Features; clear Test_Label;
clear testFiles; clear trainingSets;clear validationSets; clear TrainLabel; clear TestLabel;clear trainFiles;

clear r;
clear r1;
clear r2;
end
toc
predicted=[predictedLabels{1,1}',predictedLabels{2,1}',predictedLabels{3,1}',predictedLabels{4,1}',predictedLabels{5,1}',predictedLabels{6,1}',predictedLabels{7,1}',predictedLabels{8,1}',predictedLabels{9,1}',predictedLabels{10,1}'];
order={'check-watch', 'cross-arms', 'scratch-head', 'sit-down', 'get-up', 'turn-around', 'walk', 'wave', 'punch', 'kick',  'pick-up'};

actual=[actualLabels{1,1}',actualLabels{2,1}',actualLabels{3,1}',actualLabels{4,1}',actualLabels{5,1}',actualLabels{6,1}',actualLabels{7,1}',actualLabels{8,1}',actualLabels{9,1}',actualLabels{10,1}'];
%actual=order(actual);
cM=confusionmat((actual),(predicted));
figure
confPlot(cM,order);
GApredicted=[GApredictedLabels{1,1}',GApredictedLabels{2,1}',GApredictedLabels{3,1}',GApredictedLabels{4,1}',GApredictedLabels{5,1}',GApredictedLabels{6,1}',GApredictedLabels{7,1}',GApredictedLabels{8,1}',GApredictedLabels{9,1}',GApredictedLabels{10,1}'];
% %order={'ClimbLadder','CrawlOnKnees','DrawGraffiti','DrunkWalk','JumpOverFence','JumpOverGap','Kick','LookInCar','PickupThrowObject','PullHeavyObject','Punch','RunStop','ShotGunCollapse','SmashObject','WalkFall','WalkTurnBack','WaveArms'};
% %predicted=order(predicted);
 GAactual=[GAactualLabels{1,1}',GAactualLabels{2,1}',GAactualLabels{3,1}',GAactualLabels{4,1}',GAactualLabels{5,1}',GAactualLabels{6,1}',GAactualLabels{7,1}',GAactualLabels{8,1}',GAactualLabels{9,1}',GAactualLabels{10,1}'];
% %actual=order(actual);
 GAcM=confusionmat((GAactual),(GApredicted));
% figure
confPlot(GAcM,order);
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
% 
% %%% F-score
% f=2*1*0.967/1.967;
 F_score=2*Recall*Precision/(Precision+Recall);
 %F_score=2*1/((1/Precision)+(1/Recall));
 error=1-F_score;

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

