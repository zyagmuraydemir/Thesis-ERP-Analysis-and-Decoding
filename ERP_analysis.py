###ERP ANALYSIS###
#event-related potential analysis of epoch EEG data
#19 participants. 2 predictable & 2 random blocks per participants
#the first 3 participants (ID 1, 2, 6) have 2 predictable & 1 random blocks
#there are 4 visual stimulus types in all blocks: face, house, chair, impulse
#each epoch data has time intervals -0.1 to 0.4 seconds (501 timepoints) relative to the stimulus presentation (at 0 s)


#IMPORT
import mne
import os
import os.path
import numpy as np
import glob
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import statistics as stats
import pingouin as pg
import pandas as pd


#AGE DATA
age_list = [25, 25, 27, 22, 30, 32, 19, 26, 35, 28, 26, 33, 27, 25, 20, 22, 28, 22, 23]
#age per participant taken from written forms
avg_age = sum(age_list)/len(age_list) 
median_age = stats.median(age_list)
std_age = stats.pstdev(age_list)
del age_list


#READ EPOCHS
data_dir = '/Users/yagmu/Desktop/NewEEG'
os.chdir(data_dir)

participant_nr = list(range(1,29))
remove_pp = [3,4,5,7,10,11,12,21,27]
for na_pp in range(len(remove_pp)):
    participant_nr.remove(remove_pp[na_pp])
del remove_pp, na_pp
#all participant IDs (19) after removing missing numbers

no_ecg_pp = list(range(13,21))
no_ecg_pp.insert(0,9)
no_ecg_pp.append(22)
#10 participant IDs with 32 channels (2 channels are duplicated, 30 channels will be used), and without ECG/EOG channel

full_32_pp = list(range(23,29))
del full_32_pp[-2]
#5 participant IDs with 32 channels, ECG channel but no EOG channel

#the remaining 4 participants have 64 channels (28 channels are usable), and no ECG/EOG channel 

all_results = []
all_block_types = []
my_epochs = []
for ppnr in participant_nr:
    epoch_files = glob.glob(os.path.join(data_dir, f'pp{ppnr}_*.fif'))
    
    for each_epoch in epoch_files:
        epoch = mne.read_epochs(each_epoch, verbose=False)
        my_epochs.append(epoch)
    del each_epoch, epoch_files
    
    
    #DEFINE BLOCK TYPE (PREDICTABLE OR RANDOM)
    if ppnr<10:
        beh_dir = glob.glob(f'/Users/yagmu/Desktop/Analysis/pp0{ppnr}*')
    else:
        beh_dir = glob.glob(f'/Users/yagmu/Desktop/Analysis/pp{ppnr}*')
     
    if ppnr == 1:
        block_nr = [2,3,4]
    elif ppnr == 2 or ppnr == 6:
        block_nr = [1,2,3]
    else:
        block_nr = list(range(1,5))
    #first 3 participants (ID 1, 2, 6) don't have data from all 4 block but from only 3       
    
    block_type = [] 
    for bnr in block_nr:
        if glob.glob(beh_dir[0] + f'/ID_{ppnr}_Block_{bnr}_Sequence_*_Predictable.xlsx'):
            block_type.append('predictable')
        elif glob.glob(beh_dir[0] + f'/ID_{ppnr}_Block_{bnr}_Sequence_*_Random.xlsx'):
            block_type.append('random')
    #block_type = 4 block types for each participant (3 for first 3 participants)
    
    all_block_types.append(block_type)
    #all_block_types = block types combined for all participants (list of lists)

def flatten_matrix(matrix):
     return [item for row in matrix for item in row]
 
bt = flatten_matrix(all_block_types)
#flatten a list of lists
bt = np.array(bt)
#list to numpy array


#SEPARATING RANDOM AND PREDICTABLE BLOCKS
pred_ind = np.where(bt == 'predictable')[0]
rand_ind = np.where(bt == 'random')[0]

my_epochs_Oz = [my_epochs[i].pick('Oz') for i in (range(len(my_epochs)))]
#pick channel Oz in all blocks

pred_face_ERP = [my_epochs_Oz[i]['face'].average() for i in pred_ind]
pred_house_ERP = [my_epochs_Oz[i]['house'].average() for i in pred_ind]
pred_chair_ERP = [my_epochs_Oz[i]['chair'].average() for i in pred_ind]
pred_impulse_ERP = [my_epochs_Oz[i]['burst'].average() for i in pred_ind]
#ERPs for each condition in predictable blocks

rand_face_ERP = [my_epochs_Oz[i]['face'].average() for i in rand_ind]
rand_house_ERP = [my_epochs_Oz[i]['house'].average() for i in rand_ind]
rand_chair_ERP = [my_epochs_Oz[i]['chair'].average() for i in rand_ind]
rand_impulse_ERP = [my_epochs_Oz[i]['burst'].average() for i in rand_ind]
#ERPs for each condition in random blocks


#STANDARD ERROR OF THE MEAN (SEM)
pred_face_ERP_for_SEM = [(pred_face_ERP[i].data + pred_face_ERP[i+1].data) / 2 for i in range(0, len(pred_face_ERP), 2)]
pred_house_ERP_for_SEM = [(pred_house_ERP[i].data + pred_house_ERP[i+1].data) / 2 for i in range(0, len(pred_house_ERP), 2)]
pred_chair_ERP_for_SEM = [(pred_chair_ERP[i].data + pred_chair_ERP[i+1].data) / 2 for i in range(0, len(pred_chair_ERP), 2)]
pred_impulse_ERP_for_SEM = [(pred_impulse_ERP[i].data + pred_impulse_ERP[i+1].data) / 2 for i in range(0, len(pred_impulse_ERP), 2)]
#average ERPs of prediction blocks for each participant (2 blocks per participant)

def rand_prep(erps):
    for_SEM= erps[:3] + [(erps[i].data + erps[i+1].data) / 2 for i in range(3, len(erps), 2)]
    for_SEM[0] = for_SEM[0].data
    for_SEM[1] = for_SEM[1].data
    for_SEM[2] = for_SEM[2].data
    return for_SEM
#the first 3 participants don't have 2 random blocks but only 1
#keep the first 3 blocks, then average random blocks of the rest of the participants

rand_face_ERP_for_SEM = rand_prep(rand_face_ERP)
rand_house_ERP_for_SEM = rand_prep(rand_house_ERP)
rand_chair_ERP_for_SEM = rand_prep(rand_chair_ERP)
rand_impulse_ERP_for_SEM = rand_prep(rand_impulse_ERP)
#average ERPs of random blocks for each participant (1 or 2 blocks per participant)

def sem_erp(ERP):
    sem_temp = []
    for i in range(501):
        inner_list = []
        for j in range(len(ERP)):
            data = ERP[j]
            inner_list.append(data[0, i])
        sem_temp.append(inner_list) 
    sem = [(np.std(sem_temp[i]) / np.sqrt(len(sem_temp[i]))) for i in range(len(sem_temp))]
    return sem
#calculate SEM per timepoint

pred_face_sem = sem_erp(pred_face_ERP_for_SEM)
pred_house_sem = sem_erp(pred_house_ERP_for_SEM)
pred_chair_sem = sem_erp(pred_chair_ERP_for_SEM)
pred_impulse_sem = sem_erp(pred_impulse_ERP_for_SEM)
#SEM per timepoint in predictable blocks for each condition

rand_face_sem = sem_erp(rand_face_ERP_for_SEM)
rand_house_sem = sem_erp(rand_house_ERP_for_SEM)
rand_chair_sem = sem_erp(rand_chair_ERP_for_SEM)
rand_impulse_sem = sem_erp(rand_impulse_ERP_for_SEM)
#SEM per timepoint in random blocks for each condition

pred_face_combined = mne.evoked.combine_evoked(pred_face_ERP, weights='equal')
pred_house_combined = mne.evoked.combine_evoked(pred_house_ERP, weights='equal')
pred_chair_combined = mne.evoked.combine_evoked(pred_chair_ERP, weights='equal')
pred_impulse_combined = mne.evoked.combine_evoked(pred_impulse_ERP, weights='equal')
#average ERPS over predictable blocks for plotting

rand_face_combined = mne.evoked.combine_evoked(rand_face_ERP, weights='equal')
rand_house_combined = mne.evoked.combine_evoked(rand_house_ERP, weights='equal')
rand_chair_combined = mne.evoked.combine_evoked(rand_chair_ERP, weights='equal')
rand_impulse_combined = mne.evoked.combine_evoked(rand_impulse_ERP, weights='equal')
#average ERPS over random blocks for plotting


#PLOT PREDICTABLE ERPS WITH SEM
time_points = pred_face_combined.times
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(time_points, np.squeeze(pred_face_combined.data)* 1e6, label='Face Predictable', color='blue')
ax.plot(time_points, np.squeeze(pred_house_combined.data)* 1e6, label='House Predictable', color='orange')
ax.plot(time_points, np.squeeze(pred_chair_combined.data)* 1e6, label='Chair Predictable', color='green')
ax.plot(time_points, np.squeeze(pred_impulse_combined.data)* 1e6, label='Impulse Predictable', color='red')
#predictable ERPs plotted (all conditions combined in a single plot)
#data multiplied by 1e6 to see the amplitudes in the microvolt scale (µV)

ax.fill_between(time_points, (pred_face_combined.data.mean(axis=0) - pred_face_sem)* 1e6, (pred_face_combined.data.mean(axis=0) + pred_face_sem)* 1e6, alpha=0.2, color='blue')
ax.fill_between(time_points, (pred_house_combined.data.mean(axis=0) - pred_house_sem)* 1e6, (pred_house_combined.data.mean(axis=0) + pred_house_sem)* 1e6, alpha=0.2, color='orange')
ax.fill_between(time_points, (pred_chair_combined.data.mean(axis=0) - pred_chair_sem)* 1e6, (pred_chair_combined.data.mean(axis=0) + pred_chair_sem)* 1e6, alpha=0.2, color='green')
ax.fill_between(time_points, (pred_impulse_combined.data.mean(axis=0) - pred_impulse_sem)* 1e6, (pred_impulse_combined.data.mean(axis=0) + pred_impulse_sem)* 1e6, alpha=0.2, color='red')
#SEM per condition added

ax.set_xlabel('Time (s)')
ax.set_ylabel('Amplitude in µV')
plt.axhline(y=0, color='k', linestyle='--')
plt.axvline(x=0, color='k', linestyle='--')
ax.legend()
ax.set_title('ERPs in Predictable Condition, Channel Oz')

# currentAxis = plt.gca()
# currentAxis.add_patch(Rectangle((0.072, -2), 0.045, 8,
#                       fill=True, color='gray', alpha=0.3))
# currentAxis.add_patch(Rectangle((0.138, -2), 0.185, 8,
#                       fill=True, color='gray', alpha=0.3))
# #tinted area added to show the significant timepoints
# #based on the upcoming ANOVA results 

ax.ticklabel_format(style='plain')
plt.ylim(-2, 6)
plt.show()


#PLOT PREDICTABLE ERPS WITHOUT SEM
mne.viz.plot_compare_evokeds(
    dict(FacePredictable=pred_face_combined, HousePredictable=pred_house_combined, ChairPredictable=pred_chair_combined, ImpulsePredictable=pred_impulse_combined),
    legend="upper left",)


#PLOT RANDOM ERPS WITH SEM
time_points = rand_face_combined.times
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(time_points, np.squeeze(rand_face_combined.data)* 1e6, label='Face Random', color='blue')
ax.plot(time_points, np.squeeze(rand_house_combined.data)* 1e6, label='House Random', color='orange')
ax.plot(time_points, np.squeeze(rand_chair_combined.data)* 1e6, label='Chair Random', color='green')
ax.plot(time_points, np.squeeze(rand_impulse_combined.data)* 1e6, label='Impulse Random', color='red')
#random ERPs plotted (all conditions combined in a single plot)
#data multiplied by 1e6 to see the amplitudes in the microvolt scale (µV)

ax.fill_between(time_points, (rand_face_combined.data.mean(axis=0) - rand_face_sem)* 1e6, (rand_face_combined.data.mean(axis=0) + rand_face_sem)* 1e6, alpha=0.2, color='blue')
ax.fill_between(time_points, (rand_house_combined.data.mean(axis=0) - rand_house_sem)* 1e6, (rand_house_combined.data.mean(axis=0) + rand_house_sem)* 1e6, alpha=0.2, color='orange')
ax.fill_between(time_points, (rand_chair_combined.data.mean(axis=0) - rand_chair_sem)* 1e6, (rand_chair_combined.data.mean(axis=0) + rand_chair_sem)* 1e6, alpha=0.2, color='green')
ax.fill_between(time_points, (rand_impulse_combined.data.mean(axis=0) - rand_impulse_sem)* 1e6, (rand_impulse_combined.data.mean(axis=0) + rand_impulse_sem)* 1e6, alpha=0.2, color='red')
#SEM per condition added

ax.set_xlabel('Time (s)')
ax.set_ylabel('Amplitude in µV')
plt.axhline(y=0, color='k', linestyle='--')
plt.axvline(x=0, color='k', linestyle='--')
ax.legend(loc='upper right')
ax.set_title('ERPs in Random Condition, Channel Oz')

# currentAxis = plt.gca()
# currentAxis.add_patch(Rectangle((0.072, -2), 0.045, 8,
#                       fill=True, color='gray', alpha=0.3))
# currentAxis.add_patch(Rectangle((0.138, -2), 0.185, 8,
#                       fill=True, color='gray', alpha=0.3))
# #tinted area added to show the significant timepoints
# #based on the upcoming ANOVA results 

ax.ticklabel_format(style='plain')
plt.ylim(-2, 6)
plt.show()


#PLOT RANDOM ERPS WITHOUT SEM
mne.viz.plot_compare_evokeds(
    dict(FaceRandom=rand_face_combined, HouseRandom=rand_house_combined, ChairRandom=rand_chair_combined, ImpulseRandom=rand_impulse_combined),
    legend="upper left",)


#TWO-WAY REPEATED MEASURES ANOVA
def prepare_for_anova(erp):
    ERP_list_of_arrays = [erp[i].data for i in range(len(erp))]
    ERP_array = np.array(ERP_list_of_arrays)
    ERP_matrix = np.squeeze(ERP_array)
    return ERP_matrix
#each block type_condition_ERP variable has a list of ERPs per block (38 for predictable & 35 for random)
#transform these variables into matrices (38*501 or 35*501) 

pf_for_anova = prepare_for_anova(pred_face_ERP)
ph_for_anova = prepare_for_anova(pred_house_ERP)
pc_for_anova = prepare_for_anova(pred_chair_ERP)
pi_for_anova = prepare_for_anova(pred_impulse_ERP)
rf_for_anova = prepare_for_anova(rand_face_ERP)
rh_for_anova = prepare_for_anova(rand_house_ERP)
rc_for_anova = prepare_for_anova(rand_chair_ERP)
ri_for_anova = prepare_for_anova(rand_impulse_ERP)

all_data = np.concatenate((pf_for_anova, ph_for_anova, pc_for_anova, pi_for_anova, rf_for_anova, rh_for_anova, rc_for_anova, ri_for_anova))
#concatenate all matrices into a 292*501 matrix (38+35 blocks*4 stimulus types=292, 501 timepoints)

final_block_types = np.concatenate((np.repeat('pred', 152), np.repeat('rand', 140)))
#first 38*4=152 elements are predictable and the rest 35*4=140 elements are random
final_stimulus_types = np.concatenate((np.repeat('face', 38), np.repeat('house', 38), np.repeat('chair', 38), np.repeat('impulse', 38), np.repeat('face', 35), np.repeat('house', 35), np.repeat('chair', 35), np.repeat('impulse', 35)))
#38 each face/house/chair/impulse, then 35 each face/house/chair/impulse

pred_pp = np.repeat(np.arange(1,20), 2)
rand_pp = np.concatenate((np.array((1,2,3)), np.repeat(np.arange(4,20), 2)))
#participant indices per blocks
new_pp = np.concatenate((np.concatenate((pred_pp, pred_pp, pred_pp, pred_pp)), np.concatenate((rand_pp, rand_pp, rand_pp, rand_pp))))
#participant indices per ERPs

anovas = []
per_tp = []
for i in range(501):
    per_tp = all_data[:,i]
    dataframe_anova = pd.DataFrame({'Participant': new_pp,
        'Block Type': final_block_types,
        'Stimulus Type': final_stimulus_types,
        'ERPs': per_tp})
    anovas.append(pg.rm_anova(dv='ERPs', within=['Block Type', 'Stimulus Type'], subject='Participant',  
                      data=dataframe_anova, detailed=True))
#conduct two-way repeated measures ANOVA per timepoint (501 times)    


#MULTIPLE COMPARISON CORRECTION
p_unc_values = []
for df in anovas:
    p_unc_values.extend(df["p-unc"].values)
#find all 1053 p values (2 main effects and 1 interaction effect, therefore 3 p values per timepoint)    
fdr=mne.stats.fdr_correction(p_unc_values, alpha=0.05, method='indep')
#correct for multiple comparisons over timepoints, using Benjamini/Hochberg type FDR    


#CHECK IF THERE ARE ANY SIGNIFICANT P VALUES
def has_true_value(bool_array):
    for value in bool_array:
        if value:
            return True
    return False

result = has_true_value(fdr[0])
print(result)


#FIND THE SMALLEST P VALUE
cor_p_val = fdr[1]
min_value = min(cor_p_val)
print(min_value)


#PLOT SIGNIFICANT VALUES
block_type_main = fdr[1][np.arange(0,1501,3)]
stimulus_type_main = fdr[1][np.arange(1,1502,3)]
interaction = fdr[1][np.arange(2,1503,3)]

plt.plot(np.linspace(-0.1,0.4,501), block_type_main)
plt.axhline(y=0.05, color='r', linestyle='--', label='Threshold (0.05)')
plt.xlabel('Time (s)')
plt.ylabel('P-value')
plt.title('Block Type Main Effect')
plt.legend()
plt.show()

plt.plot(np.linspace(-0.1,0.4,501), stimulus_type_main)
plt.axhline(y=0.05, color='r', linestyle='--', label='Threshold (0.05)')
plt.xlabel('Time (s)')
plt.ylabel('P-value')
plt.title('Visual Stimulus Type Main Effect')
plt.legend()
plt.show()

plt.plot(np.linspace(-0.1,0.4,501), interaction)
plt.axhline(y=0.05, color='r', linestyle='--', label='Threshold (0.05)')
plt.xlabel('Time (s)')
plt.ylabel('P-value')
plt.title('Interaction Effect')
plt.legend()
plt.show()
    

#CHECK WHICH TIMEPOINTS ARE SIGNIFICANT
stimulus_type_main_sig = fdr[0][np.arange(1,1502,3)]
stm_sig = np.where(stimulus_type_main_sig)