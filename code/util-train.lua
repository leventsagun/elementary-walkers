function util.train(train_x, train_y)
    local epoch_size = opt.n_samples_train/opt.batch_size 
    local indices = util.permuteNfold(train_y, opt.n_workers, opt.batch_size)
    for k = 1, opt.n_steps do
        if epoch_size == 1 or k % epoch_size == 1 then -- new shuffling for every epoch or epoch_size == 1s
            indices = util.permuteNfold(train_y, opt.n_workers, opt.batch_size)
        end
        local starter = 1 
        if opt.n_workers == 0 then starter = 0 end
        for i = starter, opt.n_workers do
            local batchId = ((k - 1) % (epoch_size)) + 1
            worker[i].model:training()
            local last_layer_temp = util.cast(torch.Tensor(10):copy(worker[i].W[{{-10, -1}}]))
            local feval = function(x)
                if x ~= worker[i].W then worker[i].W:copy(x) end
                worker[i].gW:zero()
                local inputs = train_x:index(1, indices[i][batchId])
                local targets = train_y:index(1, indices[i][batchId])
                local outputs = worker[i].model:forward(inputs)
                local f = worker[i].criterion:forward(outputs, targets)
                local df_do = worker[i].criterion:backward(outputs, targets)
                worker[i].model:backward(inputs, df_do)
                return f, worker[i].gW
            end
            optim.sgd(feval, worker[i].W, worker[i].sgdState)
            worker[i].W[{{-10, -1}}]:mul(0.1)
            worker[i].W[{{-10, -1}}]:add(0.9, last_layer_temp)
            collectgarbage()
        end
        -- what to do with the center
        if opt.alpha ~= 0 then 
            util.center_averaging() 
        elseif opt.n_workers ~= 0 then 
            util.center_GD(train_x, train_y) 
        end
        -- log all info and early stopping decision
        if k % opt.freq == 0 then 
            local stop = util.logAll() 
            if stop then break end
        end
        if k % opt.freq_i == 0 then 
            util.distances_and_interpolations(k) 
        end
    end
end


function util.center_GD(train_x, train_y)
    worker[0].model:training()
    local feval = function(x)
        if x ~= worker[0].W then worker[0].W:copy(x) end
        worker[0].gW:zero()
        local outputs = worker[0].model:forward(train_x)
        local f = worker[0].criterion:forward(outputs, train_y)
        local df_do = worker[0].criterion:backward(outputs, train_y)
        worker[0].model:backward(train_x, df_do)
        return f, worker[0].gW
    end
    optim.sgd(feval, worker[0].W, worker[0].sgdState)
    collectgarbage()
end


function util.center_averaging()
    local averages = worker[0].W:clone():zero()
    local norm_averages = 0
    for i = 1, opt.n_workers do
        local diff = opt.alpha * (worker[i].W - worker[0].W)
        worker[i].W:add(-1, diff)
        norm_averages = norm_averages + worker[i].W:norm()/opt.n_workers
        averages:add(1/opt.n_workers, worker[i].W)
    end
    worker[0].W:copy(averages:div(averages:norm()/norm_averages)) --synchronious
end


function util.train_user_input(train_x, train_y)
    local trainingFlag = true
    while trainingFlag do
        io.write("continue with training (y/n)? ")
        local answer = io.read()
        if answer == '' then answer = 'y' end
        if answer == 'y' then
            io.write("enter number of steps (current " .. opt.n_steps .. "): ")
            local numSteps = io.read()
            if numSteps ~= '' then opt.n_steps = numSteps end
            io.write("enter lr (current is " .. sgdState.learningRate .. "): ")
            local lr = io.read()
            if lr ~= '' then sgdState.learningRate = lr end
            util.train(train_x, train_y)
        elseif answer == 'n' then trainingFlag = false end
    end
end


function util.explore_user_input()
    local explorationFlag = true
    local header = {{'i out of ' .. opt.n_steps_explore, 'threshold', 'step_size', 'W:norm()',
               'cost_tr', 'norm_tr', 'corr_tr', 
               'cost_te', 'norm_te', 'corr_te'}}
    util.saveCSV(header, opt.save .. '-explorations.csv')
    while explorationFlag do
        io.write("continue with exploration (y/n)? ")
        local answer = io.read()
        if answer == '' then answer = 'y' end
        if answer == 'y' then 
            io.write("enter number of steps (current " .. opt.n_steps_explore .. "): ")
            local numSteps = io.read()
            if numSteps ~= '' then opt.n_steps_explore = numSteps end
            io.write("enter threshold (current is " .. opt.threshold .. "): ")
            local th = io.read()
            if th ~= '' then opt.threshold = tonumber(th) end
            io.write("enter step_size (current is " .. opt.step_size .. "): ")
            local ss = io.read()
            if ss ~= '' then opt.step_size = tonumber(ss) end
            randomly_explore()
        elseif answer == 'n' then explorationFlag = false end
    end
end


-- -- optimize on current mini-batch
--       if opt.optimization == 'LBFGS' then

--          -- Perform LBFGS step:
--          lbfgsState = lbfgsState or {
--             maxIter = opt.maxIter,
--             lineSearch = optim.lswolfe
--          }
--          optim.lbfgs(feval, parameters, lbfgsState)
       
--          -- disp report:
--          print('LBFGS step')
--          print(' - progress in batch: ' .. t .. '/' .. dataset:size())
--          print(' - nb of iterations: ' .. lbfgsState.nIter)
--          print(' - nb of function evalutions: ' .. lbfgsState.funcEval)

--       elseif opt.optimization == 'SGD' then

--          -- Perform SGD step:
--          sgdState = sgdState or {
--             learningRate = opt.learningRate,
--             momentum = opt.momentum,
--             learningRateDecay = 5e-7
--          }
--          optim.sgd(feval, parameters, sgdState)
      
--          -- disp progress
--          xlua.progress(t, dataset:size())

--       else
--          error('unknown optimization method')
--       end