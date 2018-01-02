local py = require('fb.python')
local pd = py.import('pandas')

local CUDA_flag = false
if opt.type == 'cuda' then
    require 'cunn'
    cutorch.setDevice(opt.device_num)
    CUDA_flag = true
end


util = {}


function util.cast(t)
    if CUDA_flag then
        return t:cuda()
    else
        return t:float()
    end
end


function util.saveCSV(data, path)
    local df = pd.DataFrame(data)
    py.exec([=[df.to_csv(path, mode='a', header=None, index=False)]=], {df=df, path=path})
end


function util.model_stats(inputs, labels, id)
    worker[id].model:training()
    local costModel = util.cast(torch.Tensor(1):fill(0))
    local gradParams = util.cast(torch.Tensor(worker[id].gW:size(1)):fill(0))
    local numCorrect = torch.Tensor(1):fill(0)
    local ids = util.permuteNfold(labels, 1, 1000) -- ids is a dictionary of dictionary
    for k = 1, #ids[1] do
        worker[id].gW:zero()
        local outputs = worker[id].model:forward(inputs:index(1, ids[1][k]))
        local cost = worker[id].criterion:forward(outputs, labels:index(1, ids[1][k]))
        local grad = worker[id].criterion:backward(outputs, labels:index(1, ids[1][k]))
        worker[id].model:backward(inputs:index(1, ids[1][k]), grad)
        costModel:add(cost)
        gradParams:add(worker[id].gW)
        numCorrect:add(util.correct(outputs, util.cast(labels:index(1, ids[1][k]))))
        collectgarbage()
    end
    return costModel[1]/#ids[1], gradParams:norm()/#ids[1], numCorrect[1]
end


local header = {'i', 'opt.lr', 'evalCounter','normWeights', 'tr_cost', 'tr_normGrad', 'tr_correct', 'te_cost', 'te_normGrad', 'te_correct'}
util.saveCSV({header}, opt.save .. '-results.csv')


function util.logAll()
    local results = {}
    for i = 0, opt.n_workers do
        local tr_cost, tr_normGrad, tr_correct = util.model_stats(data['tr_x'], data['tr_y'], i) 
        local te_cost, te_normGrad, te_correct = util.model_stats(data['te_x'], data['te_y'], i)
        local normWeights = worker[i].W:norm()
        results[#results + 1] = {i, opt.lr, worker[i].sgdState.evalCounter, normWeights,
                                 tr_cost, tr_normGrad, tr_correct,
                                 te_cost, te_normGrad, te_correct}
        if not opt.silent then
            print('worker ' .. i .. ' at eval ' .. worker[i].sgdState.evalCounter, 
                  'tr:', util.round(tr_cost, 5), util.round(tr_normGrad, 5), tr_correct, 
                  'te:', util.round(te_cost, 5), util.round(te_normGrad, 5), te_correct,
                  'w:norm()', util.round(normWeights, 5))
        end
        collectgarbage()
    end
    util.saveCSV(results, opt.save .. '-results.csv')
    return util.check_stopping_condition()
end


function util.check_stopping_condition()
    if opt.stop == 'cost_value' then
        if util.mean(results, 5) < opt.stopwhen then 
            return true
        else
            return false
        end
    elseif opt.stop == 'norm_grad' then
        if util.mean(results, 6) < opt.stopwhen then 
            return true
        else
            return false
        end
    else
        return false
    end
end

------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------


function util.train_on_mid_point(initial, final)
    W:copy(find_mid_point(initial, final))
    epoch_size = opt.n_samples_train/opt.batch_size
    for k = 1, opt.n_steps_explore do 
        index = util.permuteNfold(tr_labels, 1, opt.batch_size)
        batchId = ((k - 1) % (epoch_size)) + 1
        worker[-1].model:training()
        local feval = function(x)
            if x ~= worker[-1].W then worker[-1].W:copy(x) end
            worker[-1].gW:zero()
            local inputs = tr_data:index(1, index[1][batchId])
            local targets = tr_labels:index(1, index[1][batchId])
            local outputs = worker[-1].model:forward(inputs)
            local f = worker[-1].criterion:forward(outputs, targets)
            local df_do = worker[-1].criterion:backward(outputs, targets)
            worker[-1].model:backward(inputs, df_do)
            return f, worker[-1].gW
        end
        optim.sgd(feval, worker[-1].W, worker[-1].sgdState) --sgdState.evalCounter counts every literal step
        collectgarbage()
        if k % opt.freq == 0 then
            tr_cost, tr_normGrad, tr_correct = model_stats(tr_data, tr_labels, -1) 
            te_cost, te_normGrad, te_correct = model_stats(te_data, te_labels, -1)

            local normWeights = worker[-1].W:norm()
            results = {i, opt.lr, k, normWeights,
                                     tr_cost, tr_normGrad, tr_correct,
                                     te_cost, te_normGrad, te_correct}
            print('at step ' .. k .. ' at eval ' .. worker[-1].sgdState.evalCounter, 
                  'tr:', util.round(tr_cost, 5), util.round(tr_normGrad, 5), tr_correct, 
                  'te:', util.round(te_cost, 5), util.round(te_normGrad, 5), te_correct,
                  'w:norm()', util.round(normWeights, 5)) 
            logAll(k) 
        end
        if k % opt.freq_i == 0 then log_distances_interpolations(k) end
        if stop then break end
    end 
end

