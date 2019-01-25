clear; close all;
clc;
%% data processing
M = load('D:\[2018]DL_HW1\Dataset\spam_data.mat');
%M = normc(M);
[row, col] = size(M.train_x);
data_train = M.train_x;
data_test = M.test_x;
label = M.train_y(:,1);
label1 = M.test_y(:,1);

%% initialize network

input = 40;  % depends on how many features you choose
hidden1 = 5; % second layer has 5 neuron
hidden2 = 4;
output = 1; % final layer has one neuron
l_rate = 0.0000005; % learning rate
epoch =700;
error = zeros(epoch,1);
first_layer = (randn(input,hidden1));
first_bias = (randn( 1,hidden1));
second_layer =(randn(hidden1,hidden2));
second_bias = (randn(1,hidden2));
third_layer = (randn(hidden2,output));
third_bias = (randn( 1,1));
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
            h1_act(i,j) = sigmoid(h1_output(i,j));
        end
    end

    h2_output = (h1_act*second_layer) + second_bias;
    h2_act = ones(row,hidden2);
    
    for i=1:row
        for j=1:hidden2
            h2_act(i,j) = sigmoid(h2_output(i,j));
        end
    end
    
    y_output = (h2_act*third_layer) + third_bias;
    y_act = ones(row,1);
    
    for i=1:row
        y_act(i,1) = sigmoid(y_output(i,1));
    end
    %y_act = y_output;
    %% loss function

    loss = -(label(index,1) .* log(y_act(index,1))); % +(1-label(index,1)) .* log(1-y_act(index,1))) ;
    error(k,1) = error(k,1) + mean(loss);
    %% backward pass
    %delta2
    aa = ones(row,output);
    for i=1:row
        aa(i,1) = diff_sigmoid(y_output(i,1));
    end
    
    l2 =  aa.* (label - y_act);
    
    %delta1
    aa = ones(row, hidden2);
    
    for i=1:row
        for j=1:hidden2
            aa(i,j) = diff_sigmoid(h2_output(i,j));
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
            aa(i,j) = diff_sigmoid(h1_output(i,j));
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
    disp([int2str(k),' epoch---Error : ']);
    disp(error(k,1));
end

for i=1:row
    if y_act(i,1) > 0.5
        y_act(i,1) = 1;
    else y_act(i,1) = 0;
    end
end
% x = [1:1:700];
% figure,semilogy(x,error);  % plot the learning curve
%% testing ... forward pass
h1_act = ones(row,hidden1);
row = 500;
h1_output = (data_test*first_layer) + first_bias;

for i=1:row
    for j=1:hidden1
        h1_act(i,j) = sigmoid(h1_output(i,j));
    end
end

h2_output = (h1_act*second_layer) + second_bias;
h2_act = ones(row,hidden2);
    
for i=1:row
    for j=1:hidden2
        h2_act(i,j) = sigmoid(h2_output(i,j));
    end
end
    
y_test_output = (h2_act*third_layer) + third_bias;
y_test_act = ones(row,1);
    
for i=1:row
    y_test_act(i,1) = sigmoid(y_test_output(i,1));  % test output
end

for i=1:row
    if y_test_act(i,1) > 0.5
        y_test_act(i,1) = 1;
    else y_test_act(i,1) = 0;
    end
end


%% plot the result
% x = [1:1:576];
% figure, plot(x,y_act,x,label);  % plot the training regression 
% 
% x = [1:1:192];
% figure, plot(x,y_test_act,x,label1); % plot the testing regression
% 
% loss_train = sqrt(mean((label - y_act).^2)); %the RMS error of training data
% loss_test = sqrt(mean((label1 - y_test_act).^2)); 



