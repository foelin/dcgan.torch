require 'torch'
require 'nn'
require 'cunn'
require 'optim'

opt = {
   dataset = 'lsun',       -- imagenet / lsun / folder
   batchSize = 64,
   loadSize = 96,
   fineSize = 64,
   nz = 100,               -- #  of dim for Z
   ngf = 64,               -- #  of gen filters in first conv layer
   ndf = 64,               -- #  of discrim filters in first conv layer
   nThreads = 4,           -- #  of data loading threads to use
   niter = 25,             -- #  of iter at starting learning rate
   lr = 0.0002,            -- initial learning rate for adam
   beta1 = 0.5,            -- momentum term of adam
   ntrain = math.huge,     -- #  of examples per epoch. math.huge for full dataset
   display = 1,            -- display samples while training. 0 = false
   display_id = 10,        -- display window id.
   gpu = 1,                -- gpu = 0 is CPU mode. gpu=X is GPU mode on GPU X
   name = 'gan',
   noise = 'normal',       -- uniform / normal
   resultPath = 'result',
   trainType = 1,
   port = 9000
}

-- one-line argument parser. parses enviroment variables to override the defaults
for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
opt.resultPath = paths.concat(opt.resultPath, opt.name)
print(opt)
if opt.display == 0 then opt.display = false end

opt.manualSeed = torch.random(1, 10000) -- fix seed
print("Random Seed: " .. opt.manualSeed)
torch.manualSeed(opt.manualSeed)
torch.setnumthreads(1)
torch.setdefaulttensortype('torch.FloatTensor')

-- create data loader
local DataLoader = paths.dofile('data/data.lua')
local data = DataLoader.new(opt.nThreads, opt.dataset, opt)
print("Dataset: " .. opt.dataset, " Size: ", data:size())
----------------------------------------------------------------------------
local function weights_init(m)
   local name = torch.type(m)
   if name:find('Convolution') then
      m.weight:normal(0.0, 0.02)
      m:noBias()
   elseif name:find('BatchNormalization') then
      if m.weight then m.weight:normal(1.0, 0.02) end
      if m.bias then m.bias:fill(0) end
   end
end

local nc = 3
local nz = opt.nz
local ndf = opt.ndf
local ngf = opt.ngf
local real_label = 1
local fake_label = 0

local SpatialBatchNormalization = nn.SpatialBatchNormalization
local SpatialConvolution = nn.SpatialConvolution
local SpatialFullConvolution = nn.SpatialFullConvolution

local netG = nn.Sequential()
-- input is Z, going into a convolution
netG:add(SpatialFullConvolution(nz, ngf * 8, 4, 4))
netG:add(SpatialBatchNormalization(ngf * 8)):add(nn.ReLU(true))
-- state size: (ngf*8) x 4 x 4
netG:add(SpatialFullConvolution(ngf * 8, ngf * 4, 4, 4, 2, 2, 1, 1))
netG:add(SpatialBatchNormalization(ngf * 4)):add(nn.ReLU(true))
-- state size: (ngf*4) x 8 x 8
netG:add(SpatialFullConvolution(ngf * 4, ngf * 2, 4, 4, 2, 2, 1, 1))
netG:add(SpatialBatchNormalization(ngf * 2)):add(nn.ReLU(true))
-- state size: (ngf*2) x 16 x 16
netG:add(SpatialFullConvolution(ngf * 2, ngf, 4, 4, 2, 2, 1, 1))
netG:add(SpatialBatchNormalization(ngf)):add(nn.ReLU(true))
-- state size: (ngf) x 32 x 32
netG:add(SpatialFullConvolution(ngf, nc, 4, 4, 2, 2, 1, 1))
netG:add(nn.Tanh())
-- state size: (nc) x 64 x 64

netG:apply(weights_init)

local netD = nn.Sequential()

-- input is (nc) x 64 x 64
netD:add(SpatialConvolution(nc, ndf, 4, 4, 2, 2, 1, 1))
netD:add(nn.LeakyReLU(0.2, true))
-- state size: (ndf) x 32 x 32
netD:add(SpatialConvolution(ndf, ndf * 2, 4, 4, 2, 2, 1, 1))
netD:add(SpatialBatchNormalization(ndf * 2)):add(nn.LeakyReLU(0.2, true))
-- state size: (ndf*2) x 16 x 16
netD:add(SpatialConvolution(ndf * 2, ndf * 4, 4, 4, 2, 2, 1, 1))
netD:add(SpatialBatchNormalization(ndf * 4)):add(nn.LeakyReLU(0.2, true))
-- state size: (ndf*4) x 8 x 8
netD:add(SpatialConvolution(ndf * 4, ndf * 8, 4, 4, 2, 2, 1, 1))
netD:add(SpatialBatchNormalization(ndf * 8)):add(nn.LeakyReLU(0.2, true))
-- state size: (ndf*8) x 4 x 4
netD:add(SpatialConvolution(ndf * 8, 1, 4, 4))
netD:add(nn.Sigmoid())
-- state size: 1 x 1 x 1
netD:add(nn.View(1):setNumInputDims(3))
-- state size: 1

netD:apply(weights_init)

local criterion = nn.BCECriterion()
---------------------------------------------------------------------------
optimStateG = {
   learningRate = opt.lr,
   beta1 = opt.beta1,
}
optimStateD = {
   learningRate = opt.lr,
   beta1 = opt.beta1,
}

cutorch.setDevice(opt.gpu)

----------------------------------------------------------------------------
local input = torch.CudaTensor(opt.batchSize, 3, opt.fineSize, opt.fineSize)
local noise = torch.CudaTensor(opt.batchSize, nz, 1, 1)
local label = torch.CudaTensor(opt.batchSize)
local epoch_tm = torch.Timer()
local tm = torch.Timer()
local data_tm = torch.Timer()
----------------------------------------------------------------------------

if pcall(require, 'cudnn') then
  require 'cudnn'
  cudnn.benchmark = true
  cudnn.convert(netG, cudnn)
  cudnn.convert(netD, cudnn)
end

netD:cuda();
netG:cuda();           
criterion:cuda()

local parametersD, gradParametersD = netD:getParameters()
local parametersG, gradParametersG = netG:getParameters()

if opt.display then 
  disp = require 'display' 
  disp.configure({hostname='127.0.0.1', port=opt.port})
end


noise_vis = noise:clone()
if opt.noise == 'uniform' then
    noise_vis:uniform(-1, 1)
elseif opt.noise == 'normal' then
    noise_vis:normal(0, 1)
end

paths.mkdir(opt.resultPath)

local trainIterLogger = optim.Logger(paths.concat(opt.resultPath, 'train_iter.log'))
local loss_iter = {netD={}, netG={}}
local errD = 0
local errG = 0
last_acc = 0
-- create closure to evaluate f(X) and df/dX of discriminator
local fDx = function(x)
   gradParametersD:zero()

   -- train with real
   data_tm:reset(); data_tm:resume()
   local real = data:getBatch()
   data_tm:stop()
   input:copy(real)
   label:fill(real_label)

   local output = netD:forward(input)
   local real_pred = output:float():clone()
   local errD_real = criterion:forward(output, label)
   local df_do = criterion:backward(output, label)
   netD:backward(input, df_do)

   -- train with fake
   if opt.noise == 'uniform' then -- regenerate random noise
       noise:uniform(-1, 1)
   elseif opt.noise == 'normal' then
       noise:normal(0, 1)
   end
   local fake = netG:forward(noise)
   input:copy(fake)
   label:fill(fake_label)

   local output = netD:forward(input)
   local fake_pred = output:float():clone()
   local errD_fake = criterion:forward(output, label)
   local df_do = criterion:backward(output, label)
   netD:backward(input, df_do)

   errD = (errD_real + errD_fake)/2

   -- top-1 acc
    local real_correct = real_pred:gt(0.5):sum()
    local fake_correct = fake_pred:le(0.5):sum()
    local acc = (real_correct+fake_correct)/opt.batchSize/2
    last_acc = acc
    
    print(string.format("real: %.3f, %.3f, %.3f", real_pred:max(), real_pred:min(), real_pred:mean()))
    --print(real_pred:view(1, opt.batchSize))
    print(string.format("fake: %.3f, %.3f, %.3f", fake_pred:max(), fake_pred:min(), fake_pred:mean()))
    --print(fake_pred:view(1, opt.batchSize))
    print(string.format('real: %d/%d, fake: %d/%d, acc=%.2f', real_correct, opt.batchSize, fake_correct, opt.batchSize, acc))

   return errD, gradParametersD
end



local forwardD = function()
   
   -- train with real
   data_tm:reset(); data_tm:resume()
   local real = data:getBatch()
   data_tm:stop()
   input:copy(real)
   label:fill(real_label)

   local output = netD:forward(input)
   local real_pred = output:float():clone()
   local errD_real = criterion:forward(output, label)
   
   -- train with fake
   if opt.noise == 'uniform' then -- regenerate random noise
       noise:uniform(-1, 1)
   elseif opt.noise == 'normal' then
       noise:normal(0, 1)
   end
   local fake = netG:forward(noise)
   input:copy(fake)
   label:fill(fake_label)

   local output = netD:forward(input)
   local fake_pred = output:float():clone()
   local errD_fake = criterion:forward(output, label)

   errD = (errD_real + errD_fake)/2

   -- top-1 acc
    local real_correct = real_pred:gt(0.5):sum()
    local fake_correct = fake_pred:le(0.5):sum()
    local acc = (real_correct+fake_correct)/opt.batchSize/2
    last_acc = acc
    
    print(string.format("real: %.3f, %.3f, %.3f", real_pred:max(), real_pred:min(), real_pred:mean()))
    --print(real_pred:view(1, opt.batchSize))
    print(string.format("fake: %.3f, %.3f, %.3f", fake_pred:max(), fake_pred:min(), fake_pred:mean()))
    --print(fake_pred:view(1, opt.batchSize))
    print(string.format('real: %d/%d, fake: %d/%d, acc=%.2f', real_correct, opt.batchSize, fake_correct, opt.batchSize, acc))

    return errD
end

-- create closure to evaluate f(X) and df/dX of generator
local fGx = function(x)
   gradParametersG:zero()

   --[[ the three lines below were already executed in fDx, so save computation
   noise:uniform(-1, 1) -- regenerate random noise
   local fake = netG:forward(noise)
   input:copy(fake) ]]--
   label:fill(real_label) -- fake labels are real for generator cost

   local output = netD.output -- netD:forward(input) was already executed in fDx, so save computation
   errG = criterion:forward(output, label)
   local df_do = criterion:backward(output, label)
   local df_dg = netD:updateGradInput(input, df_do)

   netG:backward(noise, df_dg)
   return errG, gradParametersG
end

local dCount=0
local gCount=0

-- train
local iter_num = 0
for epoch = 1, opt.niter do
   epoch_tm:reset()
   local counter = 0
   for i = 1, math.min(data:size(), opt.ntrain), opt.batchSize do
      tm:reset()

      iter_num = iter_num+1
      
      errG = 0
      errD = 0
     if opt.trainType == 1 then
        if last_acc < 0.8 then
          optim.adam(fDx, parametersD, optimStateD)
          dCount = dCount+1
        else
          forwardD()
          optim.adam(fGx, parametersG, optimStateG)
          gCount = gCount+1
          table.insert(loss_iter.netG, {iter_num, errG})
        end
        table.insert(loss_iter.netD, {iter_num, errD})
     else
        if last_acc < 0.8 then
          optim.adam(fDx, parametersD, optimStateD)
          optim.adam(fGx, parametersG, optimStateG)
          dCount = dCount+1
          gCount = gCount+1
        else
          forwardD()
          optim.adam(fGx, parametersG, optimStateG)
          gCount = gCount+1
        end
        table.insert(loss_iter.netD, {iter_num, errD})
        table.insert(loss_iter.netG, {iter_num, errG})
     end
       trainIterLogger:add{
        ['netG loss'] = errG,
        ['netD loss'] = errD,
      }

      -- display
      counter = counter + 1
      if counter % 10 == 0 and opt.display then
          local fake = netG:forward(noise_vis)
          local real = data:getBatch()
          disp.image(fake, {win=opt.display_id, title=opt.name})
          disp.image(real, {win=opt.display_id * 3, title=opt.name})

          disp.plot(loss_iter.netD, {win=15,title = "errD"})
          disp.plot(loss_iter.netG, {win=14,title = "errG"})
      end

      -- logging
      if ((i-1) / opt.batchSize) % 1 == 0 then
         print(('Epoch: [%d][%d / %d](%d,%d)\t Time: %.3f  DataTime: %.3f  '
                   .. '  Err_G: %.4f  Err_D: %.4f'):format(
                 epoch, ((i-1) / opt.batchSize),
                 math.floor(math.min(data:size(), opt.ntrain) / opt.batchSize),
                 dCount, gCount,
                 tm:time().real, data_tm:time().real,
                 errG, errD))
      end
   end
   
   parametersD, gradParametersD = nil, nil -- nil them to avoid spiking memory
   parametersG, gradParametersG = nil, nil
   --torch.save(paths.concat(paths.resultPath, epoch .. '_net_G.t7'), netG:clearState())
   --torch.save(paths.concat(paths.resultPath, epoch .. '_net_D.t7'), netD:clearState())
   parametersD, gradParametersD = netD:getParameters() -- reflatten the params and get them
   parametersG, gradParametersG = netG:getParameters()
   print(('End of epoch %d / %d \t Time Taken: %.3f'):format(
            epoch, opt.niter, epoch_tm:time().real))
end
