import pickle
import scipy.io as sio

with open("HighFidelity_morelli_rk45_200_2331_dnnmfbo.pkl", "rb") as f:
    data1 = pickle.load(f)

# with open("HighFidelity_stevens_euler_35.pkl", "rb") as f:
#     data2 = pickle.load(f)


# for rep_1, rep_2 in zip(data1.runs, data2.runs):
    
#     init_array_1 = [sample.cost for sample in rep_1.history[:100]]
#     init_array_2 = [sample.cost for sample in rep_2.history[:100]]

#     bo_array_1 = [sample.cost for sample in rep_1.history[100:]]
#     bo_array_2 = [sample.cost for sample in rep_2.history[100:]]
    
#     break


for rep_1 in data1.runs:
    
    init_array_1 = [sample.cost for sample in rep_1.history[:13]]
    final_array_1 = [sample.cost for sample in rep_1.history[13:]]
    # print(init_array_1)
    # print(final_array_1)
    min_c = 9999
    t = []
    for c in final_array_1:
        min_c = min(min_c, c)
        t.append(min_c)
    print(t)

print(len(t))
sio.savemat("BO_res_2331_dnnmfbo.mat", {"cost": t})