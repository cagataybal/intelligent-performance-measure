clear all

delete('C:\Users\nemesis\Desktop\RESEARCH\Intelligent_Performance_Measure\dataset\forecast_testing\realdata\*')
delete('C:\Users\nemesis\Desktop\RESEARCH\Intelligent_Performance_Measure\dataset\forecast_testing\Benchmark\*')
delete('C:\Users\nemesis\Desktop\RESEARCH\Intelligent_Performance_Measure\dataset\forecast_training\realdata\*')

imn=10; maxhid=10; 
tf='trainlm'; ep=1000; l1='tansig'; 
glo=imn*maxhid;
trratio=0.85; valratio=0; teratio=0.15; 
fcast=10; glr=1;

%%

filename='sunspot';
load(filename)
forecastlength=fcast;
lag(filename,imn,forecastlength)
load('lags.mat')

dataset=sunspot;
datasetforecast=dataset(length(dataset)-forecastlength+1:end);
dataset=dataset(1:length(dataset)-forecastlength);

    for j=1:imn
        inputs=data{j,1};
        targets=data{j,2};
    %%
            for i=1:maxhid
            net = feedforwardnet(i);

            net.trainFcn = tf;              
            net.trainParam.epochs = ep;     
            net.layers{1}.transferFcn = l1; 
            net.trainParam.showWindow = 1;  
            
%             net.divideFcn = 'divideind';
%             net.divideParam.trainInd = 1:length(dataset)-10;
%             net.divideParam.valInd = 9:12;
%             net.divideParam.testInd= length(dataset)-10:length(dataset);
             
            net.divideParam.trainRatio=trratio; 
            net.divideParam.valRatio=valratio;  
            net.divideParam.testRatio=teratio;  
            
            [net,tr] = train(net,inputs,targets);   
            outputs = net(inputs);                  
            errors = gsubtract(targets,outputs);    
            testset=targets.* tr.testMask{1};  
            testset(isnan(testset))=[];
            outputs_testset_partition=outputs.* tr.testMask{1};
            outputs_testset_partitions=outputs.* tr.testMask{1};
            outputs_testset_partition(isnan(outputs_testset_partition))=[];
            testerror = errors  .* tr.testMask{1};
            testerror(isnan(testerror))=[];

%             net_kor=corrcoef(testset,outputs_testset_partition);
%             net_corr=net_kor(1,2);

%             net_dk=abs(std(testerror)/mean(testerror));
            %% Performance measures for calculating test set error.
            MSE_testerror=mean(testerror.^2);                   
        MAE_testerror=mean(abs(testerror));                 
        MAPE_testerror=mean(abs(testerror)./abs(testset)); 
            %% Model selection algorithm for each performance measure.


                MSE_performanceError=MSE_testerror;
                global_error_MSE=MSE_testerror;
                MSE_input=j;
                MSE_hidden=i;
                MSE_inputs=inputs;
                MSE_targets=targets; MSE_tl=length(MSE_targets);
                MSE_outputs=outputs; MSE_testoutputs=outputs_testset_partition;
                MSE_testvector=MSE_targets(MSE_tl-MSE_input+1:MSE_tl);
                netbest_MSE=net;
                MSE_testset=testset;
                save(['MSE-' num2str(glr) '.mat'],'MSE_performanceError',...
                    'global_error_MSE','MSE_input','MSE_hidden','MSE_inputs',...
                    'MSE_targets','MSE_outputs','MSE_testoutputs',...
                    'MSE_testvector','netbest_MSE','MSE_testset')


        for f=1:fcast
            fc=(netbest_MSE(MSE_testvector'));
            MSE_forecast(1,f)=fc;
            MSE_fcastvector=[MSE_testvector fc];
            MSE_fcastvector=MSE_fcastvector(2:end);
            MSE_testvector=MSE_fcastvector;        
        end
        save([ 'MSE_forecast-' num2str(glr) '.mat'],'MSE_forecast')
       
                glr=glr+1;    
           end
    end
    
    
    for ar=1:glo
        load(['MSE-' num2str(ar) '.mat'])
        load(['MSE_forecast-' num2str(ar) '.mat'],'MSE_forecast')
        
        allResults{1,1}='MSE RESULTS';
        allResults{2,1}{ar,1}=MSE_input;
        allResults{2,1}{ar,2}=MSE_hidden;
        allResults{2,1}{ar,3}=1;
        allResults{2,1}{ar,4}=sqrt(MSE_performanceError);
        allResults{2,1}{ar,5}=MSE_forecast;
        allResults{2,1}{ar,6}=MSE_inputs;
        allResults{2,1}{ar,7}=MSE_targets;
        allResults{2,1}{ar,8}=MSE_testvector;
        allResults{2,1}{ar,9}=netbest_MSE;
        allResults{2,1}{ar,10}=ar;
        allResults{2,1}{ar,11}=MSE_testoutputs;
        allResults{2,1}{ar,12}=MSE_testset;
        
    end
    
    save('allResults.mat','allResults')
    assignin('base','allResults',allResults)

% load('sunspotforecast.mat')

    plot(testset,'DisplayName','Forecast','color','black');
    set(gca,'XColor', 'none','YColor','none')
    set(findall(gca, 'Type', 'Line'),'LineWidth',8);
    baseFileName3=( 'forecast_realdata.jpg');
    fullFileName3 = fullfile('C:\Users\nemesis\Desktop\RESEARCH\Intelligent_Performance_Measure\dataset\forecast_training\realdata', baseFileName3);
    saveas(gcf,fullFileName3);
    
load('allResults.mat')
glotekrar=imn*maxhid;
    
% which_dir = 'E:/Graphical_Performance_Measure/dataset/forecast_testing/Benchmark';
% dinfo = dir(which_dir);
% dinfo([dinfo.isdir]) = [];   %skip directories
% filenames = fullfile(which_dir, {dinfo.name});
% 
% which_dir = 'E:/Graphical_Performance_Measure/dataset/forecast_testing/realdata';
% dinfo = dir(which_dir);
% dinfo([dinfo.isdir]) = [];   %skip directories
% filenames = fullfile(which_dir, {dinfo.name});

for glo=1:glotekrar

    plot(allResults{2,1}{glo,11},'DisplayName','Forecast','color','black');
    set(gca,'XColor', 'none','YColor','none')
    set(findall(gca, 'Type', 'Line'),'LineWidth',8);
    baseFileName1=[ 'output_forecast-' num2str(glo) '.jpg'];
    fullFileName1 = fullfile('C:\Users\nemesis\Desktop\RESEARCH\Intelligent_Performance_Measure\dataset\forecast_testing\realdata', baseFileName1);
    saveas(gcf,fullFileName1);
    
   
    x=1:length(testset);plot(x,'color','black');
    set(gca,'XColor', 'none','YColor','none')
    set(findall(gca, 'Type', 'Line'),'LineWidth',8);
    baseFileName2=[ 'benchmark_forecast-' num2str(glo) '.jpg'];
    fullFileName2 = fullfile('C:\Users\nemesis\Desktop\RESEARCH\Intelligent_Performance_Measure\dataset\forecast_testing\Benchmark', baseFileName2);
    saveas(gcf,fullFileName2);
end

location = 'C:\Users\nemesis\Desktop\RESEARCH\Intelligent_Performance_Measure\dataset\forecast_testing\realdata\*.jpg';
ds = imageDatastore(location);

figure;
box on
perm = randperm(imn*maxhid,9);
for i = 1:9
    subplot(3,3,i);
    imshow(ds.Files{perm(i)});
    t=perm(i);
    title(['Network-',num2str(t)])
end


function lag(filename,imn,forecastlength)

% Function for creating input matrixes and target vectors.

% filename; name of the variable for time series vector in the same folder

% imn; input matrix number

datavector=cell2mat(struct2cell(load(filename)));
datavector=datavector(1:length(datavector)-forecastlength);
n=length(datavector);

for p=1:imn

for i=1:p
    
    for j=1:i
    inputvector=datavector(j:n-(p-(j-1)));
    input{1,j}=inputvector;
    
    end
    input{2,p}(i,:)=input{1,j};

end

    data{p,1}=input{2,p}; %First column of array consist of input matrixes.
    
    data{p,2}=datavector(p+1:n);%Second column of array consist of target vectors.

end

save('lags.mat','data'); %Saving the array to the realted folder.
figure
plot(datavector,'DisplayName','Forecast','color','black')
set(gca,'XColor', 'none','YColor','none')
set(findall(gca, 'Type', 'Line'),'LineWidth',5);
end