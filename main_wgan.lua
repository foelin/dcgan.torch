require 'torch'
require 'nn'
require 'optim'
require 'cunn'

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
   lr = 5e-5,            -- initial learning rate for adam
   beta1 = 0.5,            -- momentum term of adam
   ntrain = math.huge,     -- #  of examples per epoch. math.huge for full dataset
   display = 0,            -- display samples while training. 0 = false
   display_id = 10,        -- display window id.
   port = 8000,
   gpu = 1,                -- gpu = 0 is CPU mode. gpu=X is GPU mode on GPU X
   name = 'experiment1',
   noise = 'normal',       -- uniform / normal
   dIter = 5,
   clampThre = 0.01
}

-- one-line argument parser. parses enviroment variables to override the defaults
for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
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
-- state size: 1 x 1 x 1
netD:add(nn.View(1):setNumInputDims(3))
-- state size: 1

netD:apply(weights_init)


---------------------------------------------------------------------------
optimStateG = {
   learningRate = opt.lr
}
optimStateD = {
   learningRate = opt.lr
}

cutorch.setDevice(opt.gpu)

----------------------------------------------------------------------------
local input = torch.CudaTensor(opt.batchSize, 3, opt.fineSize, opt.fineSize)
local noise = torch.CudaTensor(opt.batchSize, nz, 1, 1)
local one = torch.CudaTensor(opt.batchSize):fill(1)
local mone = torch.CudaTensor(opt.batchSize):fill(-1)
local errD, errG
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


local parametersD, gradParametersD = netD:getParameters()
local parametersG, gradParametersG = netG:getParameters()

if opt.display then 
  disp = require 'display' 
  disp.configure({hostname='127.0.0.1', port=opt.port})
end

local trainEpochLogger = optim.Logger(paths.concat('checkpoints', 'train_epoch.log'))

noise_vis = noise:clone()
if opt.noise == 'uniform' then
    noise_vis:uniform(-1, 1)
elseif opt.noise == 'normal' then
    noise_vis:normal(0, 1)
end

errG = 0
errD = 0

-- create closure to evaluate f(X) and df/dX of discriminator
local fDx = function(x)
   gradParametersD:zero()
   parametersD:clamp(-opt.clampThre, opt.clampThre)

   -- train with real
   data_tm:reset();
   local real = data:getBatch()
   input:copy(real)
   
   local output = netD:forward(input)
   local errD_real = output:float():mean()
   netD:backward(input, mone)

   -- train with fake
   if opt.noise == 'uniform' then -- regenerate random noise
       noise:uniform(-1, 1)
   elseif opt.noise == 'normal' then
       noise:normal(0, 1)
   end
   local fake = netG:forward(noise)
   input:copy(fake)

   local output = netD:forward(input)
   local errD_fake = output:float():mean()
   netD:backward(input, one)

   errD = -(errD_real - errD_fake)

   return errD, gradParametersD
end

-- create closure to evaluate f(X) and df/dX of generator
local fGx = function(x)
   gradParametersG:zero()

   
   if opt.noise == 'uniform' then -- regenerate random noise
       noise:uniform(-1, 1)
   elseif opt.noise == 'normal' then
       noise:normal(0, 1)
   end
   local fake = netG:forward(noise)
   input:copy(fake) 
   
   local output = netD:forward(input)
   errG = -output:float():mean()
   
   local df_dg = netD:updateGradInput(input, mone)

   netG:backward(noise, df_dg)
   return errG, gradParametersG
end

local gTotalCount = 0

-- train
for epoch = 1, opt.niter do
   epoch_tm:reset()
   local gCount = 0
   local dCount = 0
   local dTempCount = 0  -- iters of D in in dIter

   local dIter

   local epoch_err_G = 0
   local epoch_err_D = 0
   for i = 1, math.min(data:size(), opt.ntrain), opt.batchSize do
      tm:reset()

      errG = 0
      errD = 0

      if gTotalCount < 25 or gTotalCount %500 == 0 then
        dIter = 100 
      else
        dIter = opt.dIter
      end

      if dTempCount < dIter then
        optim.rmsprop(fDx, parametersD, optimStateD)
        dTempCount = dTempCount + 1
        dCount = dCount + 1
      else
        optim.rmsprop(fGx, parametersG, optimStateG)
        dTempCount = 0
        gCount = gCount + 1
        gTotalCount = gTotalCount + 1

        -- display
        if gTotalCount % 10 == 0  and opt.display then
          local fake = netG:forward(noise_vis)
          local real = data:getBatch()
          disp.image(fake, {win=opt.display_id, title=opt.name})
          disp.image(real, {win=opt.display_id * 3, title=opt.name})
        end

      end

      

      -- logging
      if ((i-1) / opt.batchSize) % 1 == 0 then
         print(('Epoch: [%d][%d / %d](%d,%d)(%d)\t Time: %.3f  DataTime: %.3f  '
                   .. '  Err_G: %.4f  Err_D: %.4f'):format(
                 epoch, gCount+dCount,
                 math.floor(math.min(data:size(), opt.ntrain) / opt.batchSize), dTempCount, dIter, gTotalCount,
                 tm:time().real, data_tm:time().real,
                 errG, errD))


      end

      epoch_err_D = epoch_err_D + errD
      epoch_err_G = epoch_err_G + errG

   end

   trainEpochLogger:add{
    ['Err_G'] = epoch_err_G/gCount,
    ['Err_D'] = epoch_err_D/dCount,
   }

   paths.mkdir('checkpoints')
   parametersD, gradParametersD = nil, nil -- nil them to avoid spiking memory
   parametersG, gradParametersG = nil, nil
   --torch.save('checkpoints/' .. opt.name .. '_' .. epoch .. '_net_G.t7', netG:clearState())
   --torch.save('checkpoints/' .. opt.name .. '_' .. epoch .. '_net_D.t7', netD:clearState())
   parametersD, gradParametersD = netD:getParameters() -- reflatten the params and get them
   parametersG, gradParametersG = netG:getParameters()
   print(('End of epoch %d / %d \t Time Taken: %.3f'):format(
            epoch, opt.niter, epoch_tm:time().real))
end
