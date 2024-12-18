# -----------------
# GCM Perturbaitons:
# PPO
with open('drc.sh','w') as f:
    for env in ["Hopper-v3","HalfCheetah-v3","Walker2d-v3","Reacher-v2"]:
        for noise_disc in [6,10,16]:
            for prob in [0.1,0.3,0.5,0.7]:
                for seed in range(5000, 5050):
                    if env!="Hopper-v3":
                        f.write("python -m spinup.run ppo --hid \"[64,32]\" --env "+env+" --exp_name "+env+ "_gcm_d"+str(noise_disc)+"_p" +str(prob)+" --seed "+str(seed)+" --gcm_mode 1 --prob "+str(prob)+" --noise_disc "+str(noise_disc) + " --epochs 500\n")
                    else:
                        f.write("python -m spinup.run ppo --hid \"[64,32]\" --env "+env+" --exp_name "+env+ "_gcm_d"+str(noise_disc)+"_p" +str(prob)+" --seed "+str(seed)+" --gcm_mode 1 --prob "+str(prob)+" --noise_disc "+str(noise_disc) + " --epochs 500 --gamma 0.999\n")
                f.write("\n")

# RE
with open('drc.sh','a') as f:
    for env in ["Hopper-v3","HalfCheetah-v3","Walker2d-v3","Reacher-v2"]:
        for noise_disc in [6,10,16]:
            for prob in [0.1,0.3,0.5,0.7]:
                for seed in range(5000, 5050):
                    if env!="Hopper-v3":
                        f.write("python -m spinup.run re --hid \"[64,32]\" --env "+env+" --exp_name "+env+ "_gcm_d"+str(noise_disc)+"_p" +str(prob)+"_re --seed "+str(seed)+" --gcm_mode 1 --prob "+str(prob)+" --noise_disc "+str(noise_disc) + " --epochs 500\n")
                    else:
                        f.write("python -m spinup.run re --hid \"[64,32]\" --env "+env+" --exp_name "+env+ "_gcm_d"+str(noise_disc)+"_p" +str(prob)+"_re --seed "+str(seed)+" --gcm_mode 1 --prob "+str(prob)+" --noise_disc "+str(noise_disc) + " --epochs 500 --gamma 0.999\n")
                f.write("\n")

# GDRC
with open('drc.sh','a') as f:
    for env in ["Hopper-v3","HalfCheetah-v3","Walker2d-v3","Reacher-v2"]:
        for noise_disc in [6,10,16]:
            for prob in [0.1,0.3,0.5,0.7]:        
                for seed in range(5000, 5050):
                    if env=="Hopper-v3":
                        f.write("python -m spinup.run gdrc --hid \"[64,32]\" --env "+env+" --exp_name "+env+ "_gcm_d"+str(noise_disc)+"_p" +str(prob)+"_gdrc --seed "+str(seed)+" --gcm_mode 1 --prob "+str(prob)+" --noise_disc "+str(noise_disc) +" --epochs 500 --gamma 0.999\n")
                    else:
                        f.write("python -m spinup.run gdrc --hid \"[64,32]\" --env "+env+" --exp_name "+env+ "_gcm_d"+str(noise_disc)+"_p" +str(prob)+"_gdrc --seed "+str(seed)+" --gcm_mode 1 --prob "+str(prob)+" --noise_disc "+str(noise_disc) +" --epochs 500\n")
                f.write("\n")

# # SR_W
# with open('drc.sh','w') as f:
#     for env in ["Hopper-v3","HalfCheetah-v3","Walker2d-v3","Reacher-v2"]:
#         for noise_disc in [6,10,16]:
#             for prob in [0.1,0.3,0.5,0.7]:        
#                 for seed in range(5000, 5050):
#                     if env=="Hopper-v3":
#                         f.write("python -m spinup.run ppo --hid \"[64,32]\" --env "+env+" --exp_name "+env+ "_gcm_d"+str(noise_disc)+"_p" +str(prob)+"_sr_w --seed "+str(seed)+" --gcm_mode 1 --prob "+str(prob)+" --noise_disc "+str(noise_disc) +" --mode 1 --epochs 500 --gamma 0.999\n")
#                     else:
#                         f.write("python -m spinup.run ppo --hid \"[64,32]\" --env "+env+" --exp_name "+env+ "_gcm_d"+str(noise_disc)+"_p" +str(prob)+"_sr_w --seed "+str(seed)+" --gcm_mode 1 --prob "+str(prob)+" --noise_disc "+str(noise_disc) +" --mode 1 --epochs 500\n")
#                 f.write("\n")

# # DRC
# with open('drc.sh','w') as f:
#     for env in ["Hopper-v3","HalfCheetah-v3","Walker2d-v3","Reacher-v2"]:
#         for noise_disc in [6,10,16]:
#             for prob in [0.1,0.3,0.5,0.7]:
#                 for seed in range(5000, 5050):
#                     if env!="Hopper-v3":
#                         f.write("python -m spinup.run drc --hid \"[64,32]\" --env "+env+" --exp_name "+env+ "_gcm_d"+str(noise_disc)+"_p" +str(prob)+"_drc --seed "+str(seed)+" --gcm_mode 1 --prob "+str(prob)+" --noise_disc "+str(noise_disc) + " --num_outputs "+str(noise_disc)+" --epochs 500\n")
#                     else:
#                         f.write("python -m spinup.run drc --hid \"[64,32]\" --env "+env+" --exp_name "+env+ "_gcm_d"+str(noise_disc)+"_p" +str(prob)+"_drc --seed "+str(seed)+" --gcm_mode 1 --prob "+str(prob)+" --noise_disc "+str(noise_disc) + " --num_outputs "+str(noise_disc)+" --epochs 500 --gamma 0.999\n")
#                 f.write("\n")


# # -----------------
# # Continuous Perturbaitons:
# # 1st: Gaussian perturbaitons
# with open('drc.sh','w') as f:
#     for cont in [1]:
#         for env in ["Hopper-v3", "HalfCheetah-v3", "Walker2d-v3", "Reacher-v2"]:     
#                 for sigma in [1.0, 1.5, 2.0]:
#                     for seed in range(5000, 5020):
#                         if env!="Hopper-v3":
#                             f.write("python -m spinup.run ppo --hid \"[64,32]\" --env "+env+" --exp_name "+env+"_cont"+str(cont)+"_" + str(sigma)+" --seed " +str(seed)+ " --cont_mode "+str(cont)+ " --sigma "+str(sigma)+" --epochs 500 --data_dir data_ppo\n")
#                         else:
#                             f.write("python -m spinup.run ppo --hid \"[64,32]\" --env "+env+" --exp_name "+env+"_cont"+str(cont)+"_" + str(sigma)+" --seed " +str(seed)+ " --cont_mode "+str(cont)+ " --sigma "+str(sigma)+" --epochs 500 --data_dir data_ppo --gamma 0.999\n")
#                     f.write("\n")

#                     for seed in range(5000, 5020):
#                         if env!="Hopper-v3":
#                             f.write("python -m spinup.run re --hid \"[64,32]\" --env "+env+" --exp_name "+env+"_cont"+str(cont)+"_" + str(sigma)+"_re --seed " +str(seed)+ " --cont_mode "+str(cont)+ " --sigma "+str(sigma)+" --epochs 500 --data_dir data_re\n")
#                         else:
#                             f.write("python -m spinup.run re --hid \"[64,32]\" --env "+env+" --exp_name "+env+"_cont"+str(cont)+"_" + str(sigma)+"_re --seed " +str(seed)+ " --cont_mode "+str(cont)+ " --sigma "+str(sigma)+" --epochs 500 --data_dir data_re --gamma 0.999\n")
#                     f.write("\n")

#                     for seed in range(5000, 5020):
#                         if env!="Hopper-v3":
#                             f.write("python -m spinup.run gdrc --hid \"[64,32]\" --env "+env+" --exp_name "+env+"_cont"+str(cont)+"_" + str(sigma)+"_gdrc --seed " +str(seed)+ " --cont_mode "+str(cont)+ " --sigma "+str(sigma)+" --epochs 500 --data_dir data_gdrc --num_of_n_o 15\n")
#                         else:
#                             f.write("python -m spinup.run gdrc --hid \"[64,32]\" --env "+env+" --exp_name "+env+"_cont"+str(cont)+"_" + str(sigma)+"_gdrc --seed " +str(seed)+ " --cont_mode "+str(cont)+ " --sigma "+str(sigma)+" --epochs 500 --data_dir data_gdrc --num_of_n_o 15 --gamma 0.999\n")
#                     f.write("\n")

# # 2nd and 3rd: uniform and reward range uniform perturbaitons
# with open('drc.sh','w') as f:
#     for cont in [2, 3]:
#         for env in ["Hopper-v3", "HalfCheetah-v3", "Walker2d-v3", "Reacher-v2"]:
#             for sigma in [0.1, 0.2, 0.3, 0.4]:
#                 for seed in range(5000, 5020):
#                     if env!="Hopper-v3":
#                         f.write("python -m spinup.run ppo --hid \"[64,32]\" --env "+env+" --exp_name "+env+"_cont"+str(cont)+"_" + str(sigma)+" --seed " +str(seed)+ " --cont_mode "+str(cont)+ " --sigma "+str(sigma)+" --epochs 500 --data_dir data_ppo\n")
#                     else:
#                         f.write("python -m spinup.run ppo --hid \"[64,32]\" --env "+env+" --exp_name "+env+"_cont"+str(cont)+"_" + str(sigma)+" --seed " +str(seed)+ " --cont_mode "+str(cont)+ " --sigma "+str(sigma)+" --epochs 500 --data_dir data_ppo --gamma 0.999\n")
#                 f.write("\n")

#                 for seed in range(5000, 5020):
#                     if env!="Hopper-v3":
#                         f.write("python -m spinup.run re --hid \"[64,32]\" --env "+env+" --exp_name "+env+"_cont"+str(cont)+"_" + str(sigma)+"_re --seed " +str(seed)+ " --cont_mode "+str(cont)+ " --sigma "+str(sigma)+" --epochs 500 --data_dir data_re\n")
#                     else:
#                         f.write("python -m spinup.run re --hid \"[64,32]\" --env "+env+" --exp_name "+env+"_cont"+str(cont)+"_" + str(sigma)+"_re --seed " +str(seed)+ " --cont_mode "+str(cont)+ " --sigma "+str(sigma)+" --epochs 500 --data_dir data_re --gamma 0.999\n")
#                 f.write("\n")

#                 for seed in range(5000, 5020):
#                     if env!="Hopper-v3":
#                         f.write("python -m spinup.run gdrc --hid \"[64,32]\" --env "+env+" --exp_name "+env+"_cont"+str(cont)+"_" + str(sigma)+"_gdrc --seed " +str(seed)+ " --cont_mode "+str(cont)+ " --sigma "+str(sigma)+" --epochs 500 --data_dir data_gdrc --num_of_n_o 15\n")
#                     else:
#                         f.write("python -m spinup.run gdrc --hid \"[64,32]\" --env "+env+" --exp_name "+env+"_cont"+str(cont)+"_" + str(sigma)+"_gdrc --seed " +str(seed)+ " --cont_mode "+str(cont)+ " --sigma "+str(sigma)+" --epochs 500 --data_dir data_gdrc --num_of_n_o 15 --gamma 0.999\n")
#                 f.write("\n")