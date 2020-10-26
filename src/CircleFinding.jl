module CircleFinding

using CircleFit
using DelimitedFiles
using Plots
using StatsBase
using Random
using Statistics

function read_profiles(csv_file)
    a = readdlm(csv_file,',')
    b = map(v -> typeof(v) == Int64 ? v / 1000.0 : NaN,a)[:,2:end]
    x = b[:,1:2:end]
    y = b[:,2:2:end]
    return (x,y)
end

"""

"""
function circle_ransac(data; radius = 0.98, threshold = 0.2, max_iters = 100, number_points = 8)
    data_length = size(data)[2]
    best_xc = 0
    best_yc = 0
    best_rfit = 0;
    best_err = 10000000;
    for i = 1:max_iters
        ri = randperm(data_length)
        maybe_in = data[:,ri[1:3]]
        xc,yc,rfit = circfit(maybe_in)
        also_in = hcat(filter(xy -> abs(sqrt((xy[1]-xc)Z^2 + (xy[2]-yc)^2) - rfit) < threshold, eachcol(data[:,ri[4:end]]))...)

        if size(also_in)[2] > number_points && abs(rfit - radius) < threshold
            current_data = hcat(maybe_in,also_in)
            better_xc, better_yc, better_rfit = circfit(current_data)

            err = circle_error(current_data,better_xc,better_yc,better_rfit)

            if err < best_err 
                best_xc = better_xc
                best_yc = better_yc
                best_rfit = better_rfit
                best_err = err
            end
        end
    end
    return (best_xc, best_yc, best_rfit, best_err)
end

circle_error(data, xc, yc, radius) = sqrt(mean((radius - sqrt.(sum((data .- [xc;yc]).^2,dims=1))).^2))

function fit_circle(x1,y1,x2,y2,radius)
    x3 = (x1+x2) ./ 2
    y3 = (y1+y2) ./ 2
    q = sqrt((x1-x2)^2 + (y1-y2)^2)

    xc = x3 - sqrt(radius^2 - (q/2)^2) * (y1 - y2)/q
    yc = y3 - sqrt(radius^2 - (q/2)^2) * (x2 - x2)/q
    return(xc,yc)
end
    
end