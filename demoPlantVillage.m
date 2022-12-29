imgSets = imageSet('E:\Coding\plantvillage dataset\color', 'recursive'); m=1;srcTrainFiles=[];Trainlabels=[];
srcTestFiles=[];Testlabels=[];TrainFeatures=[];TestFeatures=[];
[trainset, testset] = partition(imgSets, 0.7,'randomized');
%gpuDevice(1)

net=alexnet;

layer ='fc6';
for i=1:length(trainset) srcTrainFiles=dir(strcat('E:\Coding\plantvillage dataset\color\',trainset(i).Description,'\*.jpg'));
    for j=1:length(srcTrainFiles)
        srcTrainFiles(j).name = strcat('E:\Coding\plantvillage dataset\color\',trainset(i).Description,'\',srcTrainFiles(j).name);
         %r=regexp(srcTrainFiles(j).name,'\','split');
        img=imread(srcTrainFiles(j).name);%dirData(i).name
        ff =activations(net,img,layer);
        ff=ff(:)';
        TrainFeatures=[TrainFeatures;ff];
        clear img
    l=length(srcTrainFiles);
    Trainlabel=str2num(trainset(i).Description);
    Trainlabels=[Trainlabels;Trainlabel];
    %Trainlabels=sourceFiles; 
    m=m+1;   
    end
 end
clear i;clear j;clear m;
m=1;
for i=1:length(testset) srcTestFiles=dir(strcat('E:\Coding\plantvillage dataset\color\',testset(i).Description,'\*.jpg'));
    for j=1:length(srcTestFiles)
        srcTestFiles(j).name = strcat('E:\Coding\plantvillage dataset\color\',testset(i).Description,'\',srcTestFiles(j).name);
        img=imread(srcTestFiles(j).name);%dirData(i).name
        fff =activations(net,img,layer);
        fff=fff(:)';
        TestFeatures=[TestFeatures;fff];
        clear img
    
    l=length(srcTestFiles);
    Testlabel=str2num(testset(i).Description);
    %TestsourceFiles=[srcTestFiles;srcTestFiles];
    Testlabels=[Testlabels;Testlabel]; 
    m=m+1;
    end
 end
% g=strcat('E:\Coding\features_Plant.mat');
% 
% save(g,'Trainlabels','Testlabels','trainingFeatures','testingFeatures');
% 
trainlabel = (Trainlabels);
testlabel = (Testlabels);
traindata=TrainFeatures';%permute(trainingFeatures,[3 2 1]);%(traindata);%
testdata=TestFeatures';%permute(testFeatures,[3 2 1]); %(testdata); %
nn.label = trainlabel;

[trainlabel, testlabel] = label_convertMU(trainlabel,testlabel,'2');
%-----------Setting----------------------------------------------------
method            = {'ELM','RELM'};

nn.hiddensize     = 10000;
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


[nn, acc_train,label_actual,label_desired]   = elm_train(traindata, trainlabel, nn);
[nn1, acc_test,label_actual_test,label_desired_test]    = elm_test(testdata, testlabel, nn);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                   Evolutionary deep features
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
FitVal1=acc_test;
NumGeneration=100;
[GATable, OptFeaturesTrain,OptFeaturesTest,location,maxFitness]=EvolutionaryEDTMD(NumGeneration,FitVal1,traindata,testdata,trainlabel,testlabel);%figure
[nn, GAacc_train,GAlabel_actual,GAlabel_desired]   = elm_train(OptFeaturesTrain, trainlabel, nn);
[nn1, GAacc_test,GAlabel_actual_test,GAlabel_desired_test]    = elm_test(OptFeaturesTest, testlabel, nn);
GAactualLabels{j,:}=GAlabel_actual_test';
GApredictedLabels{j,:}=(GAlabel_desired_test)';
g=strcat('E:\Coding\Genetic\AugmentationCode-20210921T094000Z-001\AugmentationCode\GACNN_',int2str(j),'.mat');

save(g,'OptFeaturesTrain','OptFeaturesTest','location','maxFitness','GAactualLabels','GApredictedLabels');
