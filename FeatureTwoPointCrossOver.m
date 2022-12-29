%%
%               Random crossover Twopoints
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function stateChromosome=FeatureTwoPointCrossOver(mdl,Features,labels)
[r c]=size(Features)
UpdatedFeatures=zeros(r, c);
point1=randi([1,round(r/2)-1]);
point2=randi([round(r/2)+1,r-1]);

%point=3;
    for j=1:r
        selectedRow=setdiff([1:j],randi([1,r-1]));
    [chromosome] = Crossover2Point(Features(j,:), Features(selectedRow,:), point1,point2);
    UpdatedFeatures(j,:)=chromosome;
   
    
    label_desired_r_test=predict(mdl,UpdatedFeatures(j,:));%transposee

    label_actual_r_test=labels(1,j);
%UpdatedFeatures= UpdatedFeatures;
%confMatrix=confusionmat(label_actual_r_test,label_desired_r_test);
     numcorrect =(label_actual_r_test==label_desired_r_test)
     size(numcorrect)
     FitVal(1,j)= (numcorrect)%/length(label_actual_r_test));
    end
%accuracy = mean(accuracy);

%% Intial fitness value
%UpdatedFeatures

 % f= size(FitVal)
  % l=size(labels')
   %uf=size(UpdatedFeatures)
 stateChromosome=table(FitVal',UpdatedFeatures,labels');
    
end
%trainIndex=setdiff([1:N],z);