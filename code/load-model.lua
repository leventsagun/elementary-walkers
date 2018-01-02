-- continue with the load model from load_data.lua

require 'nn'


function load.feed_forward()
    local model = nn.Sequential() 
    model:add(nn.Reshape(data['input_size']))
    model:add(nn.Linear(data['input_size'], opt.n_hidden))
    model:add(nn.ReLU())
    for i = 2, opt.n_layers do 
        model:add(nn.Linear(opt.n_hidden, opt.n_hidden))
        model:add(nn.ReLU())
    end
    model:add(nn.Linear(opt.n_hidden, 10))
    model:add(nn.LogSoftMax())
    local criterion = nn.ClassNLLCriterion()
    return util.cast(model), util.cast(criterion)
end


function load.feed_forward_hinton()
    local model = nn.Sequential() 
    model:add(nn.Reshape(data['input_size']))
    model:add(nn.Linear(data['input_size'], opt.n_hidden))
    model:add(nn.ReLU())
    for i = 2, opt.n_layers do 
        model:add(nn.Linear(opt.n_hidden, opt.n_hidden))
        model:add(nn.ReLU())
    end
    model:add(nn.Linear(opt.n_hidden, 10))
    model:add(nn.LogSoftMax())
    dofile('crit-hintonize_NLL.lua') 
    -- there are two alternative versions
    local criterion = nn.Hinton_2()
    return util.cast(model), util.cast(criterion)
end


function load.lenet()
    local model = nn.Sequential() 
    model:add(nn.SpatialConvolution(3,16,5,5))
    model:add(nn.ReLU())
    model:add(nn.SpatialMaxPooling(2,2,2,2))
    model:add(nn.SpatialConvolution(16,16,5,5))
    model:add(nn.ReLU())
    model:add(nn.SpatialMaxPooling(2,2,2,2))
    model:add(nn.View(16 * 5 * 5):setNumInputDims(3))
    model:add(nn.Linear(16 * 5 * 5, 10))
    local criterion = nn.CrossEntropyCriterion()
    return util.cast(model), util.cast(criterion)
end


function load.initialize()
    worker = {}
    worker[-1] = {}
    if opt.model == 'nn' then
        worker[-1].model, worker[-1].criterion = load.feed_forward()
    elseif opt.model == 'hinton' then
        worker[-1].model, worker[-1].criterion = load.feed_forward_hinton()
    elseif opt.model == 'lenet' then
        worker[-1].model, worker[-1].criterion = load.lenet()
    end
    worker[-1].W, worker[-1].gW = worker[-1].model:getParameters()
    worker[-1].sgdState = {
        learningRate = opt.lr, 
        momentum = opt.mom
    }
    worker[-1].sgdState.evalCounter = 0
    for i = 0, opt.n_workers do
        worker[i] = {}
        worker[i].model = worker[-1].model:clone()
        worker[i].criterion = worker[-1].criterion:clone()
        worker[i].W, worker[i].gW = worker[i].model:getParameters()
        worker[i].sgdState = util.shallowcopy(worker[-1].sgdState)        
        if opt.weight_init_std > 0 then 
            worker[i].W:normal(0, opt.weight_init_std) 
        end
    end
end

load.initialize() -- initialize all models

