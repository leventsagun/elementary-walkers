
function util.sharpness(c_init, c_final)
    return 100*(c_final - c_init)/(1 + c_init)
end


function util.local_max(worker_id, epsilon)
    local bs = opt.n_samples_train
    worker[-1].W:copy(worker[worker_id].W)
    local cost_tr_init, _, __ = util.model_stats(data['tr_x'], data['tr_y'], worker_id)
    local cost_te_init, _, __ = util.model_stats(data['te_x'], data['te_y'], worker_id)
    local center = worker[worker_id].W:clone()
    local epoch_size = opt.n_samples_train/bs 
    local indices = util.permuteNfold(data['tr_y'], 1, bs)
    worker[-1].model:training()
    local k = 0
    while true do
        k = k + 1
        local batchId = ((k - 1) % (epoch_size)) + 1
        local feval = function(x)
            --worker[-1].W:copy(x)
            worker[-1].gW:zero()
            local inputs = data['tr_x']:index(1, indices[1][batchId])
            local targets = data['tr_y']:index(1, indices[1][batchId])
            local outputs = worker[-1].model:forward(inputs)
            local f = worker[-1].criterion:forward(outputs, targets)
            local df_do = worker[-1].criterion:backward(outputs, targets)
            worker[-1].model:backward(inputs, df_do)
            return f, -worker[-1].gW
        end
        optim.sgd(feval, worker[-1].W, {learningRate = 0.0001})

        local cost_tr, norm_tr, corr_tr = util.model_stats(data['tr_x'], data['tr_y'], -1)
        local cost_te, norm_te, corr_te = util.model_stats(data['te_x'], data['te_y'], -1)
        local diff = (worker[-1].W - worker[worker_id].W):norm()
        local sharp = {}
        sharp['tr'] = util.sharpness(cost_tr_init, cost_tr)
        sharp['te'] = util.sharpness(cost_te_init, cost_te) 
        util.saveCSV({sharp}, opt.save .. '-sharpness.csv')
        if diff > epsilon then 
            if not opt.silent then
                print('tr:', util.round(cost_tr, 5), util.round(norm_tr, 5), corr_tr,
                    'te:', util.round(cost_te, 5), util.round(norm_te, 5), corr_te,
                    'dist&sharpness', util.round(diff, 5), util.round(sharp['tr'], 5), util.round(sharp['te'], 5))
                print('number of steps for sharpness calculation', k)
                util.saveCSV({{k, k}}, opt.save .. '-sharpness.csv')
            end
            break 
        end
        if k > 1000 then
            print('takes too long, change threshold')
            break
        end
    end
end



