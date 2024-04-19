    load('sunspotforecast.mat')
    dataforecast=sunspotforecast;
    load('allResults.mat')
    imn=10; maxhid=10; glotekrar=imn*maxhid;
    
which_dir = 'C:\Users\nmss\Desktop\Convolutional_Neural_Networks\dataset\forecast_testing\Benchmark';
dinfo = dir(which_dir);
dinfo([dinfo.isdir]) = [];   %skip directories
filenames = fullfile(which_dir, {dinfo.name});
delete( filenames{:} )

which_dir = 'C:\Users\nmss\Desktop\Convolutional_Neural_Networks\dataset\forecast_testing\realdata';
dinfo = dir(which_dir);
dinfo([dinfo.isdir]) = [];   %skip directories
filenames = fullfile(which_dir, {dinfo.name});
delete( filenames{:} )
    
for glo=1:glotekrar
%     plot(dataforecast,'DisplayName','dataforecast','color','black');hold on;
    plot(allResults{2,1}{glo,5},'DisplayName','mseforecast','color','black');hold on;
%     plot(allResults{2,4}{glo,5},'DisplayName','mapeforecast');
%     plot(allResults{2,5}{glo,5},'DisplayName','xerrorforecast');
    hold off;
    set(gca,'XColor', 'none','YColor','none')
    set(findall(gca, 'Type', 'Line'),'LineWidth',8);
    baseFileName1=[ 'output_forecast-' num2str(glo) '.jpg'];
    fullFileName1 = fullfile('C:\Users\nmss\Desktop\Convolutional_Neural_Networks\dataset\forecast_testing\realdata', baseFileName1);
    saveas(gcf,fullFileName1);
    x=1:10;plot(x,'color','black');
    set(gca,'XColor', 'none','YColor','none')
    set(findall(gca, 'Type', 'Line'),'LineWidth',8);
    baseFileName2=[ 'benchmark_forecast-' num2str(glo) '.jpg'];
    fullFileName2 = fullfile('C:\Users\nmss\Desktop\Convolutional_Neural_Networks\dataset\forecast_testing\Benchmark', baseFileName2);
    saveas(gcf,fullFileName2);
end


    load('sunspot.mat')
    datatestprediction=sunspot;
    load('allResults.mat')
for glo=1:100
%     plot(datatestprediction(length(datatestprediction)-length(allResults{2,1}{glo,11})+1:end),'DisplayName','datatestprediction');hold on;
    plot(allResults{2,1}{glo,11},'DisplayName','mseforecast');hold on;
%     plot(allResults{2,4}{glo,11},'DisplayName','mapeforecast');
%     plot(allResults{2,5}{glo,11},'DisplayName','xerrorforecast');
    hold off;
    saveas(gcf,[ 'output_data-' num2str(glo) '.jpg']);
end

% regresyon bencmark !!!

