function [outputArg1] = diff_sigmoid(h_out)
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here
    outputArg1 = sigmoid(h_out) * (1 - sigmoid(h_out));
end

