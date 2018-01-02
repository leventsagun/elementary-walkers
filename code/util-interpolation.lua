-- save headers for the interpolations and distances
local header = {{'i', 'j', 'l', 'k', 'params[l]:norm()', 'cost_tr', 'norm_tr', 'corr_tr', 'cost_te', 'norm_te', 'corr_te'}}
util.saveCSV(header, opt.save .. '-interpolations.csv')
local header = {}
for i = 0, opt.n_workers do
    for j = 0, opt.n_workers do
        header[#header + 1] = j .. '_to_' .. i
    end
end
util.saveCSV({header}, opt.save .. '-distances.csv')


function util.distances_and_interpolations(k)
    -- evaluate linear interpolations among workers/interpolation on the shell
    local distances = torch.zeros(opt.n_workers + 1, opt.n_workers + 1)
    for i = 0, opt.n_workers do 
        for j = 0, i - 1 do
            distances[i + 1][j + 1] = (worker[i].W - worker[j].W):norm()
            if not opt.silent then 
                print('from worker ' .. j .. ' to worker ' .. i) 
            end
            -- vector of interpolation coefficients
            -- torch.linspace(opt.resolution*opt.c1, opt.resolution*opt.c2, opt.resolution*(opt.c2-opt.c1))
            for l = opt.resolution*opt.c1, opt.resolution*opt.c2 do
                local params = util.spherical_convex_combination(worker[j].W, worker[i].W, l/opt.resolution)
                worker[-1].W:copy(params)
                local cost_tr, norm_tr, corr_tr = util.model_stats(data['tr_x'], data['tr_y'], -1)
                local cost_te, norm_te, corr_te = util.model_stats(data['te_x'], data['te_y'], -1)
                local interpolations = {i, j, l, k, params:norm(),
                                     cost_tr, norm_tr, corr_tr,
                                     cost_te, norm_te, corr_te}
                util.saveCSV({interpolations}, opt.save .. '-interpolations.csv')
                if not opt.silent then
                    print('tr:', util.round(cost_tr, 5), util.round(norm_tr, 5), corr_tr,
                          'te:', util.round(cost_te, 5), util.round(norm_te, 5), corr_te,
                          'W:norm()',util.round(params:norm(), 5))
                end
            end
        end
    end
    local dists = torch.add(distances, distances:transpose(1, 2))
    util.saveCSV(torch.totable(dists:reshape(1, (opt.n_workers + 1)^2)), opt.save .. '-distances.csv')
    if not opt.silent and opt.n_workers ~= 0 then 
        print(dists) 
    end
    collectgarbage()
end
