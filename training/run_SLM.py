import os

# 
bs = 16  # Batch size
lr = 4e-3 # Learning rate
num_epochs = 100  # Training epochs
diffusers_per_epoch = 10  # Number of scattering media used per training epoch
test_diffusers_per_epoch = 10  # Number of scattering media used per test epoch

z1_multipliers = [0.1] # Distance between the object and the scattering medium
# z1_multipliers = [0.1, 0.2, 0.3, 0.4]



for z1_multiplier in z1_multipliers:
    expid = "Z1-" + str(z1_multiplier) + "_2w"    # file name
    
    cmd = "python DynaDiffuser_2layer_v2_SLM.py --expid {} --num_epochs {} --z1_multiplier {} --bs {} --lr {} --diffusers_per_epoch {} --test_diffusers_per_epoch {}".format(expid, num_epochs, z1_multiplier, bs, lr, diffusers_per_epoch, test_diffusers_per_epoch)

    print('running cmd: {}'.format(cmd))
    os.system(cmd)

