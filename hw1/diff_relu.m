function [outputArg1] = diff_relu(inputArg1)
%UNTITLED7 Summary of this function goes here
%   Detailed explanation goes here
    if inputArg1>0
        outputArg1 = 1;
    elseif inputArg1==0
        outputArg1 = 0.5;
    else
        outputArg1 = 0;
    end
end

