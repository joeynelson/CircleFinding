module CircleFinding

using CircleFit
using DelimitedFiles
using Plots
using StatsBase
using Random
using Statistics
using Distributions
using Optim
using ForwardDiff

function read_profiles(csv_file)
    a = readdlm(csv_file,',')
    b = map(v -> typeof(v) == Int64 ? v / 1000.0 : NaN,a)[:,2:end]
    x = b[:,1:2:end]
    y = b[:,2:2:end]
    return (x,y)
end

"""

"""
function circle_ransac(data, radius; threshold = 0.4, max_iters = 2000, number_points = 8)
    data_length = size(data)[2]
    best_xc = 0
    best_yc = 0
    best_rfit = 0;
    best_err = 10000000;
    for i = 1:max_iters
        ri = randperm(data_length)
        maybe_in = data[:,ri[1:3]]
        xc,yc,rfit = circfit(maybe_in)
        also_in = hcat(filter(xy -> abs(sqrt((xy[1]-xc)^2 + (xy[2]-yc)^2) - radius) < threshold, collect(eachcol(data[:,ri[4:end]])))...)

        if size(also_in)[1] == 2 && size(also_in)[2] > number_points && abs(rfit - radius) < threshold
            current_data = hcat(maybe_in,also_in)
            better_xc, better_yc, better_rfit = circfit(current_data)

            err = circle_error(current_data,better_xc,better_yc,radius)

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

circfit(xy) = circfit(xy[1,:],xy[2,:])

circle_error(data, xc, yc, radius) = sqrt(mean((radius .- sqrt.(sum((data .- [xc;yc]).^2,dims=1))).^2))

function filterdata(x,y)
    data = vcat(x[:]',y[:]')
    return hcat(filter(xy -> isfinite(xy[1]) && isfinite(xy[2]),collect(eachcol(data)))...)
end

function fit_circle(x1,y1,x2,y2,radius)
    x3 = (x1+x2) ./ 2
    y3 = (y1+y2) ./ 2
    q = sqrt((x1-x2)^2 + (y1-y2)^2)

    xc = x3 - sqrt(radius^2 - (q/2)^2) * (y1 - y2)/q
    yc = y3 - sqrt(radius^2 - (q/2)^2) * (x2 - x2)/q
    return(xc,yc)
end

function circle_hough_map(data,radius; xbins = (-15,15,500), ybins = (0, 30, 500))
    bins = zeros(ybins[3],xbins[3])
    step_size = (xbins[2]-xbins[1]) / (xbins[3]-1)

    bx = LinRange(xbins...)
    by = LinRange(ybins...)

    d = Normal(radius, step_size)
    for xy = eachcol(data)
        xstart = findindex(xy[1] - radius - step_size, xbins)
        xend = findindex(xy[1] + radius + step_size, xbins)
        ystart = findindex(xy[2] - radius - step_size, ybins)
        yend = findindex(xy[2] + step_size, ybins)
        bins[ystart:yend,xstart:xend] .+= [pdf(d, sqrt((xy[1] - x)^2 + (xy[2] - y)^2)) for y = by[ystart:yend], x = bx[xstart:xend]]
    end
    return bins
end

function circle_hough(data,radius; xbins = (-15,15,500), ybins = (0, 30, 500))

    bx = LinRange(xbins...)
    by = LinRange(ybins...)

    bins = circle_hough_map(data, radius, xbins=xbins, ybins=ybins)
    max_tuple = findmax(bins)
    return (bx[max_tuple[2][2]], by[max_tuple[2][1]],max_tuple[1])
end

function circle_hybrid(data,radius)
    x,y,w = circle_hough(data,radius)

    good_data = hcat(filter(xy -> radius-0.1 < sqrt( (xy[1] - x)^2 + (xy[2] - y)^2) < radius+0.1,collect(eachcol(data)))...)'
    cir_fit_err(p, x, y) = sum((sqrt.((x .- p[1]).^2 .+ (y .- p[2]).^2) .- p[3]).^2)
    
    res = optimize(b -> cir_fit_err(b,good_data[:,1],good_data[:,2]),[x,y,radius], LBFGS(), autodiff=:forward)
    return vcat(Optim.minimizer(res),length(good_data),Optim.minimum(res))
end


function findindex(x,bins)
    step_size = (bins[2]-bins[1]) / (bins[3]-1)
    index = round(Integer, (x - bins[1]) / step_size) + 1
    return clamp(index, 1, bins[3])
end

    
end