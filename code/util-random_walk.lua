if opt.explore then
    local header = {{'for the worker ', 'threshold', 'step_size', 'W:norm()',
                     'cost_tr', 'norm_tr', 'corr_tr', 
                     'cost_te', 'norm_te', 'corr_te'}}
    util.saveCSV(header, opt.save .. '-explorations.csv')
end


function util.random_walker(worker_id)
    local cost_init, _, __ = util.model_stats(data['tr_x'], data['tr_y'], worker_id)
    local epsilon = 0.01
    opt.threshold = cost_init + epsilon
    worker[-1].W:copy(worker[worker_id].W)
    while true do
        local step = torch.Tensor(worker[-1].W:size()[1]):normal():mul(opt.rw_step_size)
        worker[-1].W:add(util.cast(step))
        local cost_tr, norm_tr, corr_tr = util.model_stats(data['tr_x'], data['tr_y'], -1)
        local cost_te, norm_te, corr_te = util.model_stats(data['te_x'], data['te_y'], -1)
        if not opt.silent then
            print('tr:', util.round(cost_tr, 5), util.round(norm_tr, 5), corr_tr,
                  'te:', util.round(cost_te, 5), util.round(norm_te, 5), corr_te,
                  'W:norm()', util.round(worker[-1].W:norm(), 5))
        end
        local rw = {worker_id, opt.threshold, opt.rw_step_size, worker[-1].W:norm(),
                    cost_tr, norm_tr, corr_tr,
                    cost_te, norm_te, corr_te} 
        util.saveCSV({rw}, opt.save .. '-explorations.csv')
        if cost_tr - cost_init > opt.threshold then 
            worker[worker_id].W:copy(worker[-1].W)
            break 
        end
    end
end

