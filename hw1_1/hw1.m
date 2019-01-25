clear; close all;
clc;
%% data processing
M = csvread('D:\[2018]DL_HW1\Dataset\energy_efficiency_data.csv',1,0);
%M = normc(M);
[row, col] = size(M(:,:));
%% ALL FEATURE
m1 = M(:,6);
a = ones(row,4);
for i=1:row  % one hot encoding
    if(m1(i)==2)
        a(i,:) = a(i,:) .* [1,0,0,0];
    end
    if(m1(i)==3)
        a(i,:) = a(i,:) .* [0,1,0,0];
    end
    if(m1(i)==4)
        a(i,:) = a(i,:) .* [0,0,1,0];
    end
    if(m1(i)==5)
        a(i,:) = a(i,:) .* [0,0,0,1];
    end
end

m2 = M(:,8);
a2 = ones(row,6);
for i=1:row  % one-hot encoding
    if(m2(i)==0)
        a2(i,:) = a2(i,:) .* [1,0,0,0,0,0];
    end
    if(m2(i)==1)
        a2(i,:) = a2(i,:) .* [0,1,0,0,0,0];
    end
    if(m2(i)==2)
        a2(i,:) = a2(i,:) .* [0,0,1,0,0,0];
    end
    if(m2(i)==3)
        a2(i,:) = a2(i,:) .* [0,0,0,1,0,0];
    end
    if(m2(i)==4)
        a2(i,:) = a2(i,:) .* [0,0,0,0,1,0];
    end
    if(m2(i)==5)
        a2(i,:) = a2(i,:) .* [0,0,0,0,0,1];
    end
end

data = zeros(row,15);
data(:,1) = M(:,1);
data(:,2) = M(:,2);
data(:,3) = M(:,4);
data(:,4) = M(:,7);
data(:,5) = M(:,5);
data(:,6:9) = a;
data(:,10:15) = a2;
data_test = data(577:768,:);
data_train = data(1:576,:);
[row, col] = size(data_train);

label = M(:,9);
label = label(1:576,1);
label1 = M(:,9);
label1 = label1(577:768,1);
%% 3 KEY FEATURE
% m1 = M(:,3);
% m2 = M(:,4);
% m3 = M(:,7);
% data = zeros(row,3);
% data(:,1) = m1;
% data(:,2) = m2;
% data(:,3) = m3;
% data_train = data(1:576,:);
% data_test = data(577:768,:);
% 
% label = M(:,9);
% label = label(1:576,1);
% label1 = M(:,9);
% label1 = label1(577:768,1);
% 
% row = 576;

%% initialize network

input = 15;  % depends on how many features you choose
hidden1 = 5; % second layer has 5 neuron
hidden2 = 4;
output = 1; % final layer has one neuron
l_rate = 0.000000005; % learning rate
epoch =50;
error = zeros(epoch,1);
first_layer = abs(randn(input,hidden1));
first_bias = abs(randn( 1,hidden1));
second_layer = abs(randn(hidden1,hidden2));
second_bias = abs(randn(1,hidden2));
third_layer = abs(randn(hidden2,output));
third_bias = abs(randn( 1,1));
%% gradient discent
for k=1:epoch
%     if mod(k,100) == 0    % learning rate can decay every 100 epoch
%         l_rate = l_rate/10;    
%     end
    %SGD
    for index=1:row 
    %% forward pass
    h1_act = ones(row,hidden1);

    h1_output = (data_train*first_layer) + first_bias;

    for i=1:row
        for j=1:hidden1
            h1_act(i,j) = relu(h1_output(i,j));
        end
    end

    h2_output = (h1_act*second_layer) + second_bias;
    h2_act = ones(row,hidden2);
    
    for i=1:row
        for j=1:hidden2
            h2_act(i,j) = relu(h2_output(i,j));
        end
    end
    
    y_output = (h2_act*third_layer) + third_bias;
    y_act = ones(row,1);
    
    for i=1:row
        y_act(i,1) = relu(y_output(i,1));
    end
    %y_act = y_output;
    %% loss function

    loss = (label - y_act).^2;
    error(k,1) = error(k,1) + mean(loss);
    %% backward pass
    %delta2
    aa = ones(row,output);
    for i=1:row
        aa(i,1) = diff_relu(y_output(i,1));
    end
    
    l2 =  aa.*(2 .* (-label + y_act));
    
    %delta1
    aa = ones(row, hidden2);
    
    for i=1:row
        for j=1:hidden2
            aa(i,j) = diff_relu(h2_output(i,j));
        end
    end
    
    l1_all = ones(row,1,hidden2);

    for i=1:row
        for j=1:hidden2
            l1_all(i,1,j) = aa(i,j) .* third_layer(j,1) .* l2(i,1);
        end
    end
    l1 = ones(row,hidden2);
    
    for i=1:hidden2
        l1(:,i) = l1_all(:,1,i);
    end
    
    %delta0
    aa = ones(row, hidden1);
    
    for i=1:row
        for j=1:hidden1
            aa(i,j) = diff_relu(h1_output(i,j));
        end
    end
    
    l0_all = ones(row,1,hidden1);

    for i=1:row
        for j=1:hidden1
            l0_all(i,1,j) = aa(i,j) .* (sum(second_layer(j,:) .* l1(i,:)));  
        end
    end
    
    l0 = ones(row,hidden1);
    
    for i=1:hidden1
        l0(:,i) = l0_all(:,1,i);
    end
    
     %% update parameter
    %for i=1:row
    third_layer = third_layer - l_rate .* ((h2_act(index,:).').*l2(index,1));
    third_bias = third_bias - l_rate .* (l2(index,1));
    for j=1:hidden2
        second_layer(:,j) = second_layer(:,j) - l_rate .* ((h1_act(index,:).').*l1(index,j));
        second_bias = second_bias - l_rate .* (l1(index,:));
    end
        
    for j=1:hidden1
        first_layer(:,j) = first_layer(:,j) - l_rate .* (data_train(index,:).'*l0(index,j));
        first_bias = first_bias - l_rate .* (l0(index,:));
    end
end
    disp([int2str(k),' epoch']), disp(error(k));
end
x = [1:1:epoch];
figure,semilogy(x,error);  % plot the learning curve
title('log(error)'); 
%% testing ... forward pass
h1_act = ones(row,hidden1);
row = 192;
h1_output = (data_test*first_layer) + first_bias;

for i=1:row
    for j=1:hidden1
        h1_act(i,j) = relu(h1_output(i,j));
    end
end

h2_output = (h1_act*second_layer) + second_bias;
h2_act = ones(row,hidden2);
    
for i=1:row
    for j=1:hidden2
        h2_act(i,j) = relu(h2_output(i,j));
    end
end
    
y_test_output = (h2_act*third_layer) + third_bias;
y_test_act = ones(row,1);
    
for i=1:row
    y_test_act(i,1) = relu(y_test_output(i,1));  % test output
end
%% plot the result
x = [1:1:576];
figure, plot(x,y_act,x,label);  % plot the training regression 
title('train regression');
x = [1:1:192];
figure, plot(x,y_test_act,x,label1); % plot the testing regression
title('test regression');
loss_train = sqrt(mean((label - y_act).^2)); %the RMS error of training data
loss_test = sqrt(mean((label1 - y_test_act).^2)); 

disp(['loss train = ', num2str(loss_train)]);
disp(['loss test = ', num2str(loss_test)]);

