function lag(filename,imn)

% Function for creating input matrixes and target vectors.

% filename; name of the variable for time series vector in the same folder

% imn; input matrix number

datavector=cell2mat(struct2cell(load(filename)));
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
plot(datavector)
end