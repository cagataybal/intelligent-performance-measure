
bestmodelindex=16

figure
forecasts=allResults{2,1}{bestmodelindex,11};
testset=allResults{2,1}{bestmodelindex,12};
plot(forecasts,'DisplayName','TestPrediction','color','red');
hold on;
% figure
plot(testset,'DisplayName','TestSet','color','black');
hold off;
legend

MAE_testerror=mean(abs(allResults{2,1}{bestmodelindex,11}-allResults{2,1}{bestmodelindex,12}))
MAPE_testerror=mean(abs(allResults{2,1}{bestmodelindex,11}-allResults{2,1}{bestmodelindex,12})./abs(allResults{2,1}{bestmodelindex,12}))
RMSE_testerror=sqrt(mean((allResults{2,1}{bestmodelindex,11}-allResults{2,1}{bestmodelindex,12}).*(allResults{2,1}{bestmodelindex,11}-allResults{2,1}{bestmodelindex,12})))

corrcoef(forecasts,testset)

da1_rmse=forecasts;
da2_rmse=da1_rmse(2:length(da1_rmse))-da1_rmse(1:length(da1_rmse)-1);
da3_rmse=testset;
da4_rmse=da3_rmse(2:length(da3_rmse))-da3_rmse(1:length(da3_rmse)-1);
da5_rmse=da2_rmse.*da4_rmse;
da6_rmse=da5_rmse >= 0;
DA_test=1-mean(da6_rmse)


% set(gca,'XColor', 'none','YColor','none')
% set(findall(gca, 'Type', 'Line'),'LineWidth',8);

%% FORECAST PLOTS

figure
forecasts=allResults{2,1}{bestmodelindex,5};
plot(forecasts,'DisplayName','Forecast','color','red');
hold on;
% figure
plot(antimonyforecast,'DisplayName','Validation','color','black');
hold off;
legend

% set(gca,'XColor', 'none','YColor','none')
% set(findall(gca, 'Type', 'Line'),'LineWidth',8);


%%
bestforecast=(forecasts-min(forecasts))/(max(forecasts)-(min(forecasts)));
normdataforecast=(datasetforecast-min(datasetforecast))/(max(datasetforecast)-min(datasetforecast));

figure
plot(bestforecast,'DisplayName','Forecast','color','red');
hold on;
plot(normdataforecast,'DisplayName','Validation','color','black');
hold off;
set(gca,'XColor', 'none','YColor','none')
set(findall(gca, 'Type', 'Line'),'LineWidth',8);

% figure
% plot(allResults{2,1}{53,11},'DisplayName','Test Prediction','color','red');
% hold on;
% plot(allResults{2,1}{53,12},'DisplayName','Test Set','color','black');
% hold off;
% set(gca,'XColor', 'none','YColor','none')
% set(findall(gca, 'Type', 'Line'),'LineWidth',8);
