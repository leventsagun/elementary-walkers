function util.save_metadata()
    local keys={}
    local values={}
    for key, value in pairs(opt) do
        keys[#keys + 1] = key
        values[#values + 1] = value
    end
    util.saveCSV({keys}, opt.save .. '-metadata.csv')
    util.saveCSV({values}, opt.save .. '-metadata.csv')
end
util.save_metadata()


function util.convex_combination(initial, final, alpha)
    local cx_comb = util.cast(torch.zeros(initial:size()[1]))
    cx_comb:add(alpha, final):add(1-alpha, initial)    
    return util.cast(cx_comb)
end


function util.spherical_convex_combination(initial, final, alpha)
    local cx_comb = util.cast(torch.zeros(initial:size()[1]))
    cx_comb:add(alpha, final):add(1-alpha, initial)
    cx_comb:div(cx_comb:norm()/(alpha*final:norm() + (1 - alpha)*initial:norm()))
    return util.cast(cx_comb)
end


function util.permuteNfold(list, N, splitSize)
    local idxs = {}
    for i = 0, N do -- for every worker it gives a different shuffling
        idxs[i] = torch.randperm(list:size(1)):long():split(splitSize)
    end
    return idxs
end


function util.hintonize(labels, p, q)
    local len = labels:size(1)
    local targets = torch.Tensor(len, 10):fill(p)
    for i = 1, len do
        targets[i]:narrow(1, labels[i], 1):fill(q)
    end
    return targets
end


function util.shallowcopy(orig)
    local orig_type = type(orig)
    local copy
    if orig_type == 'table' then
        copy = {}
        for orig_key, orig_value in pairs(orig) do
            copy[orig_key] = orig_value
        end
    else -- number, string, boolean, etc
        copy = orig
    end
    return copy
end


function util.mean(t, column)
    -- http://lua-users.org/wiki/SimpleStats
    local sum = 0
    local count= 0
    for k, v in pairs(t) do
        sum = sum + v[column]
        count = count + 1
    end
    return sum/count
end


function util.normalize(data)
    data:add(-data:mean())
    data:mul(1/data:std())
    return data
end


function util.correct(output, label)
    local labels = torch.zeros(label:size(1))
    local _
    if opt.model == 'hinton' then
        _, labels = torch.max(label, 2)
    else
        labels = label
    end
    local _, preds = torch.max(output, 2)
    return torch.sum(torch.eq(util.cast(labels), util.cast(preds)))
end


function util.round(num, idp)
    --http://lua-users.org/wiki/SimpleRound
    if num then return tonumber(string.format("%." .. (idp or 0) .. "f", num)) end
    if not num then return 'NA' end
end

