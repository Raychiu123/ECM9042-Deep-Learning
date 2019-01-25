function [outputArg1] = relu(inputArg1)
%UNTITLED6 Summary of this function goes here
%   Detailed explanation goes here
    if inputArg1 >=0
        outputArg1 = inputArg1;
    else outputArg1 = 0 ;  %.01*inputArg1;
    end
end

