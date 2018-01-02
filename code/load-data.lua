-- data should be in dict format with tr_x, tr_y, te_x, te_y

load = {}

function load.mnist()
    local py = require('fb.python')
    py.exec("execfile('load-data.py')")
    data = py.eval('load_mnist()')
    data['tr_y']:add(1)
    data['val_y']:add(1)
    data['te_y'] = util.cast(data['te_y']:add(1))
    data['te_x'] = util.cast(data['te_x'])
    --concat train and val...
    data['tr_x'] = util.cast(torch.cat(data['tr_x'], data['val_x'], 1))
    data['tr_y'] = util.cast(torch.cat(data['tr_y'], data['val_y'], 1))
    data['input_size'] = 28*28
end


function load.play_mnist()
    if opt.mnist_type == 'double' then
        data['input_size'] = 2*28*28
        local idx_tr = torch.randperm(data['tr_x']:size(1)):long()
        local idx_te = torch.randperm(data['te_x']:size(1)):long()
        -- concat them together
        data['tr_x'] = data['tr_x']:cat(data['tr_x']:index(1, idx_tr))
        data['te_x'] = data['te_x']:cat(data['te_x']:index(1, idx_te))
        data['tr_y'] = (data['tr_y'] + data['tr_y']:index(1, idx_te) - 2) % 10 + 1
        data['te_y'] = (data['te_y'] + data['te_y']:index(1, idx_te) - 2) % 10 + 1
    elseif opt.mnist_type == 'scrambled' then
        local idx = torch.randperm(data['tr_y']:size(1)):long()
        data['tr_y'] = data['tr_y']:index(1, idx)
    elseif opt.mnist_type == 'all_noise' then
        data['tr_x'] = data['tr_x']:normal()
    elseif opt.mnist_type == 'some_noise' then
        local k = 7
        local lines = torch.randperm(28)[{{1, k}}]:long()
        local lines_ = torch.randperm(28)[{{1, k}}]:long()
        for i = 1, k do
            data['tr_x']:select(2, lines[i]):normal()
            data['tr_x']:select(3, lines_[i]):normal()            
        end
    end
end


function load.cifar10()
    local Provider = torch.class 'Provider'
    local provider = torch.load('../../provider.t7')
    data = {}
    data['input_size'] = 3*32*32
    data['tr_x'] = util.cast(provider.trainData.data:float())
    data['te_x'] = util.cast(provider.testData.data:float())
    data['tr_y'] = util.cast(provider.trainData.labels)
    data['te_y'] = util.cast(provider.testData.labels)
end


if opt.data == 'mnist' then 
    load.mnist() 
end

if opt.model == 'hinton' then 
    data['tr_y'] = util.cast(util.hintonize(data['tr_y'], opt.q, opt.p))
    data['te_y'] = util.cast(util.hintonize(data['te_y'], 0, 1))
end

if opt.mnist_type ~= 'regular' then 
    load.play_mnist() 
end

if opt.data == 'cifar10' then 
    load.cifar10() 
end

if opt.n_samples_train < data['tr_x']:size(1) then
    local ind_tr = torch.randperm(data['tr_x']:size(1)):long()
    data['tr_x'] = data['tr_x']:index(1, ind_tr)[{{1, opt.n_samples_train}}]
    data['tr_y'] = data['tr_y']:index(1, ind_tr)[{{1, opt.n_samples_train}}]
end

