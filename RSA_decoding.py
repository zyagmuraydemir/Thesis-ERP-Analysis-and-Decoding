###DECODING WITH REPRESENTATIONAL SIMILARITY ANALYSIS###
#representational similarity analysis with mahalanobis distance and leave-one-trial-out cross validation
#19 participants. 2 predictable & 2 random blocks per participants
#the first 3 participants (ID 1, 2, 6) have 2 predictable & 1 random blocks
#the same 2 block types are concatenated per participant
#there are 4 visual stimulus types in all blocks: face, house, chair, impulse
#decoding up to 3 stimulus labels preceding the impulse images separately in random and predictable blocks
#each epoch data has time intervals -0.1 to 0.4 seconds (501 timepoints) relative to the stimulus presentation (at 0 s)
#each block has 1125 trials (some of them are dropped so usually it is fewer)
#around 10 percent of the trials have the impulse image, which will be used for decoding
#there are 6 datasets/conditions: block type (random/predictable) * position (n-1/n-2/n-3)
#the datasets are shaped ~225 trials * ~30 channels * 501 timepoints
#dimensionality reduction with both F statistics and signal to noise ratio (SNR) is conducted to have 5 channels at the end
#so the final shape is ~225 trials * 5 channels * 501 timepoints per dataset per participant
#trial labels are decoded separately for timepoint/participant from selected 5 channels
#~225 trials include a balanced distribution (~75 each) of 3 stimulus types (face, house, chair)


#IMPORT
import mne
import os
import os.path
import numpy as np
import glob
import matplotlib.pyplot as plt
import rsatoolbox
from rsatoolbox.rdm import calc_rdm_unbalanced
from rsatoolbox.rdm.rdms import concat
import pandas as pd
import scipy.stats as stats
import pingouin as pg


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
#save RSA results
all_block_types = []
#save block types (random or predictable)
chan_rand = []
chan_pred = []
#save which 5 channels were used per participant/condition
for ppnr in participant_nr:
    epoch_files = glob.glob(os.path.join(data_dir, f'pp{ppnr}_*.fif'))
    
    my_epochs = []
    for each_epoch in epoch_files:
        epoch = mne.read_epochs(each_epoch, verbose=False)
        my_epochs.append(epoch)
    del each_epoch, epoch, epoch_files
    
    
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
    #all_block_types = block types combined for all participants


    #UPDATE ANNOTATIONS AND INDICES AFTER DROPPING BAD TRIAL  
    concat_rand_n1 = []
    concat_rand_n2 = []
    concat_rand_n3 = []
    concat_pred_n1 = []
    concat_pred_n2 = []
    concat_pred_n3 = []
    #save normalized data of each position/block type
    
    concat_stim_rand_n1 = []
    concat_stim_rand_n2 = []
    concat_stim_rand_n3 = []
    concat_stim_pred_n1 = []
    concat_stim_pred_n2 = []
    concat_stim_pred_n3 = []
    #save corrected annotations (stimulus labels) of each position/block type
    
    concat_rand_face = []
    concat_rand_house = []
    concat_rand_chair = []
    concat_pred_face = []
    concat_pred_house = []
    concat_pred_chair = []
    #save different stimulus epochs for dimensionality reduction (F statistics) in each block type
    
    concat_snr_rand = []
    concat_snr_pred = []
    #save different block types for dimensionality reduction (signal to noise ratio)
    
    for block in range(len(block_nr)):
        remaining_ind_og = my_epochs[block].selection
        #trial indices after dropping bad & upside down trials (original scale, 0-1124)
        my_annotations_og = my_epochs[block].annotations.description
        #1125 annotations (original)
        true_annotations = my_annotations_og[remaining_ind_og].tolist()
        #remaining annotations after dropping bad trials
        true_stimuli = np.column_stack((true_annotations, remaining_ind_og))
        #matched remaining annotations and remaining indices (original scale, 0-1124)
        del my_annotations_og


        #FIND CORRECTED INDICES OF IMPULSE, N-1, N-2, AND N-3 TRIALS
        impulse_ind_new = [i for i, annotation in enumerate(true_annotations) if annotation == 'burst']
        #corrected new indices of the impulse trials
        
        for k in range(1,4):
            vars()[f'n{k}_ind_new'] = [np.where(remaining_ind_og == (remaining_ind_og[i] - k))[0][0] for i in impulse_ind_new if np.isin(remaining_ind_og[i] - k, remaining_ind_og)]
            #corrected new indices of the n-1/n-2/n-3 trials
            #after this line, I only use corrected new indices for impulse, n-1, n-2, and n-3
        
        
            #GET IMPULSE INDICES FOR EACH POSITION
            vars()[f'n{k}_impulse_ind_new'] = [i for i in impulse_ind_new if np.isin(remaining_ind_og[i] - k, remaining_ind_og)]
            #I will use the impulse trials' data to decode 3 preceding trials' labels
            #after finding the corrected preceding incides that are not dropped in the last step, I need to find impulse indices that correspond to those preceding indices
        

            #FIND CORRECTED STIMULUS TYPES IN N-1, N-2, AND N-3
            vars()[f'n{k}_stim'] = [true_annotations[i] for i in vars()[f'n{k}_ind_new']]
            vars()[f'n{k}_true'] = np.column_stack((vars()[f'n{k}_stim'], vars()[f'n{k}_ind_new']))


            #COMBINE CORRECTED PRECEDING STIMULUS TYPES AND IMPULSE INDICES
            vars()[f'n{k}_stim_imp'] = np.stack((vars()[f'n{k}_stim'], vars()[f'n{k}_impulse_ind_new']), axis=1)

    
        #CONCATENATE SAME BLOCK TYPES PER PARTICIPANT (FOR DECODING & COVARIANCE MATRICES)
        n1_data=my_epochs[block].get_data(picks='eeg', item=n1_impulse_ind_new)
        n2_data=my_epochs[block].get_data(picks='eeg', item=n2_impulse_ind_new)
        n3_data=my_epochs[block].get_data(picks='eeg', item=n3_impulse_ind_new)
        
        
        #NORMALIZE DATA
        sd_n1_data = np.std(n1_data[:,:,:100], axis=2)
        sd_rep = np.repeat(sd_n1_data[:, :, np.newaxis], 501, axis=2)
        n1_data = np.divide(n1_data, sd_rep)
        
        sd_n2_data = np.std(n2_data[:,:,:100], axis=2)
        sd_rep = np.repeat(sd_n2_data[:, :, np.newaxis], 501, axis=2)
        n2_data = np.divide(n2_data, sd_rep)
        
        sd_n3_data = np.std(n3_data[:,:,:100], axis=2)
        sd_rep = np.repeat(sd_n3_data[:, :, np.newaxis], 501, axis=2)
        n3_data = np.divide(n3_data, sd_rep)
        #used baseline data (0-100 ms) for normalizing
        
        if block_type[block] == 'random':
            concat_rand_n1.append(n1_data)
            concat_rand_n2.append(n2_data)
            concat_rand_n3.append(n3_data)
            
            concat_stim_rand_n1.extend(n1_stim)
            concat_stim_rand_n2.extend(n2_stim)
            concat_stim_rand_n3.extend(n3_stim)
            
        elif block_type[block] == 'predictable': 
            concat_pred_n1.append(n1_data)
            concat_pred_n2.append(n2_data)
            concat_pred_n3.append(n3_data)
            
            concat_stim_pred_n1.extend(n1_stim)
            concat_stim_pred_n2.extend(n2_stim)
            concat_stim_pred_n3.extend(n3_stim) 
           
    
        #CONCATENATE SAME BLOCK TYPES PER PARTICIPANT (FOR F STATS DIMENSIONALITY REDUCTION)     
        face_all_ind = np.where(true_stimuli[:, 0] == 'face')[0]
        house_all_ind = np.where(true_stimuli[:, 0] == 'house')[0]
        chair_all_ind = np.where(true_stimuli[:, 0] == 'chair')[0]
        
        face_all_data = my_epochs[block].get_data(picks='eeg', item=face_all_ind)
        house_all_data = my_epochs[block].get_data(picks='eeg', item=house_all_ind)
        chair_all_data = my_epochs[block].get_data(picks='eeg', item=chair_all_ind)
        
        if block_type[block] == 'random':
            concat_rand_face.append(face_all_data)
            concat_rand_house.append(house_all_data)
            concat_rand_chair.append(chair_all_data)
        elif block_type[block] == 'predictable':
            concat_pred_face.append(face_all_data)
            concat_pred_house.append(house_all_data)
            concat_pred_chair.append(chair_all_data)
            
        
        #CONCATENATE SAME BLOCK TYPES PER PARTICIPANT (FOR SNR DIMENSIONALITY REDUCTION)     
        for_snr = my_epochs[block].get_data(picks='eeg')
        
        if block_type[block] == 'random':
            concat_snr_rand.append(for_snr)
        elif block_type[block] == 'predictable':
            concat_snr_pred.append(for_snr)

    
    #FIX THE FORMAT OF THE CONCATENATED VARIABLES
    concat_rand_n1 = np.concatenate(concat_rand_n1, axis=0)
    concat_rand_n2 = np.concatenate(concat_rand_n2, axis=0)
    concat_rand_n3 = np.concatenate(concat_rand_n3, axis=0)
    concat_pred_n1 = np.concatenate(concat_pred_n1, axis=0)
    concat_pred_n2 = np.concatenate(concat_pred_n2, axis=0)
    concat_pred_n3 = np.concatenate(concat_pred_n3, axis=0)
    
    concat_rand_face = np.concatenate(concat_rand_face, axis=0)
    concat_rand_house = np.concatenate(concat_rand_house, axis=0)
    concat_rand_chair = np.concatenate(concat_rand_chair, axis=0)
    concat_pred_face = np.concatenate(concat_pred_face, axis=0)
    concat_pred_house = np.concatenate(concat_pred_house, axis=0)
    concat_pred_chair = np.concatenate(concat_pred_chair, axis=0)
    
    concat_snr_rand = np.concatenate(concat_snr_rand, axis=0)
    concat_snr_pred = np.concatenate(concat_snr_pred, axis=0)


    #SNR & F STATS FOR DIMENSIONALITY REDUCTION
    snr_rand_mean = np.nanmean(concat_snr_rand, axis=0)
    snr_pred_mean = np.nanmean(concat_snr_pred, axis=0)
    #mean of all predictable and random trials of a participant (channel number * timepoints)
    
    signal_rand = []
    for row in snr_rand_mean:
        squared_values = np.square(row[150:251])
        mean_squared = np.mean(squared_values)
        signal_rand.append(np.sqrt(mean_squared))
    signal_rand = np.array(signal_rand)
        
    signal_pred = []
    for row in snr_pred_mean:
        squared_values = np.square(row[150:251])
        mean_squared = np.mean(squared_values)
        signal_pred.append(np.sqrt(mean_squared))
    signal_pred = np.array(signal_pred)
    #the root square mean of the first 50 to 150 milliseconds post-stimulus for each channel per block type
    #one signal value per channel for each block type per participant
    
    noise_rand = []
    for row in snr_rand_mean:
        squared_values = np.square(row[:101])
        mean_squared = np.mean(squared_values)
        noise_rand.append(np.sqrt(mean_squared))
    noise_rand = np.array(noise_rand)
        
    noise_pred = []
    for row in snr_pred_mean:
        squared_values = np.square(row[:101])
        mean_squared = np.mean(squared_values)
        noise_pred.append(np.sqrt(mean_squared))
    noise_pred = np.array(noise_pred)
    #the root square mean of the baseline (0 to 100 milliseconds) for each channel per block type
    #one noise value per channel for each block type per participant
        
    snr_rand_result = np.square(signal_rand / noise_rand)
    snr_pred_result = np.square(signal_pred / noise_pred)
    #divided signal to noise per channel, then squared
    #one SNR value per channel and per participant for each block type
        
    snr_log_rand = 10 * np.log10(snr_rand_result)
    snr_log_pred = 10 * np.log10(snr_pred_result)
    #converted to the logarithmic decibel scale by multiplying the base 10 logarithm of the SNR value by 10
        
    snr_threshold = 5
    snr_rand_ind = np.where(snr_log_rand > snr_threshold)[0]
    snr_pred_ind = np.where(snr_log_pred > snr_threshold)[0]
    #channels with a value higher than the threshold of 5 dB were selected as a preliminary list for dimensionality reduction
    
    f_stat_rand = mne.stats.f_oneway(concat_rand_face, concat_rand_house, concat_rand_chair)
    #one-way ANOVA (for random trials) per participant to compare visual stimulus types
    #all trials of a participant (except for impulse trials) were used
    #the groups compared: Face/Predictable, House/Predictable, and Chair/Predictable
    #result: F value per channel and timepoint
    mean_f_stat_rand = np.mean(f_stat_rand, axis=1)
    #F-values were averaged over timepoints, resulting in one F value per channel
    snr_met_rand = mean_f_stat_rand[snr_rand_ind]
    #channels that met the SNR criterion were ordered by F value 
    temp_sort = np.argsort(snr_met_rand)[-5:]
    #then the 5 channels with highest F values were selected
    temp_chan_ind = np.sort(temp_sort)
    ind_chan_select_rand = snr_rand_ind[temp_chan_ind] 
    
    f_stat_pred = mne.stats.f_oneway(concat_pred_face, concat_pred_house, concat_pred_chair)
    #one-way ANOVA (for predictable trials) per participant to compare visual stimulus types
    #all trials of a participant (except for impulse trials) were used
    #the groups compared: Face/Random, House/Random, and Chair/Random 
    #result: F value per channel and timepoint
    mean_f_stat_pred = np.mean(f_stat_pred, axis=1)
    #F-values were averaged over timepoints, resulting in one F value per channel
    snr_met_pred = mean_f_stat_pred[snr_pred_ind]
    #channels that met the SNR criterion were ordered by F value 
    temp_sort = np.argsort(snr_met_pred)[-5:]
    #then the 5 channels with highest F values were selected
    temp_chan_ind = np.sort(temp_sort)
    ind_chan_select_pred = snr_pred_ind[temp_chan_ind] 
    
    
    #GET SELECTED CHANNEL DATA
    n1_data_rand = concat_rand_n1[:, ind_chan_select_rand, :]
    n2_data_rand = concat_rand_n2[:, ind_chan_select_rand, :]
    n3_data_rand = concat_rand_n3[:, ind_chan_select_rand, :]
    
    n1_data_pred = concat_pred_n1[:, ind_chan_select_pred, :]
    n2_data_pred = concat_pred_n2[:, ind_chan_select_pred, :]
    n3_data_pred = concat_pred_n3[:, ind_chan_select_pred, :]
    
    del concat_rand_face, concat_rand_house, concat_rand_chair, concat_pred_face, concat_pred_house, concat_pred_chair
        
    all_chan_names=my_epochs[0].info.ch_names
    #get all channel names
    
    chan_sel_by_nr=[]
    for i in range(ind_chan_select_rand.size):
        if ppnr not in no_ecg_pp and ppnr not in full_32_pp:
            if ind_chan_select_rand[i]>29:
                chan_sel_by_nr.append(ind_chan_select_rand[i]+2)
            else:
                chan_sel_by_nr.append(ind_chan_select_rand[i])
        else:
            chan_sel_by_nr.append(ind_chan_select_rand[i])    
    true_channels_rand=[all_chan_names[i] for i in chan_sel_by_nr]
    #find selected channel names for concatenated random data

    chan_sel_by_nr=[]
    for i in range(ind_chan_select_pred.size):
        if ppnr not in no_ecg_pp and ppnr not in full_32_pp:
            if ind_chan_select_pred[i]>29:
                chan_sel_by_nr.append(ind_chan_select_pred[i]+2)
            else:
                chan_sel_by_nr.append(ind_chan_select_pred[i])
        else:
            chan_sel_by_nr.append(ind_chan_select_pred[i])    
    true_channels_pred=[all_chan_names[i] for i in chan_sel_by_nr]
    #find selected channel names for concatenated predictable data

    chan_rand.append(true_channels_rand)
    chan_pred.append(true_channels_pred)
    #add them to the general list of selected channels
    
    
    #DATASET FOR RSA & COVARIANCE MATRICES 
    t = np.arange(-0.1, 0.4 + 1/1000, 1/1000)
    #timepoints as miliseconds
    
    n1_rand_dataset = rsatoolbox.data.TemporalDataset(n1_data_rand, descriptors = {'subj': ppnr}, channel_descriptors={'channels': true_channels_rand}, obs_descriptors={'conds': concat_stim_rand_n1}, time_descriptors={'time': t})
    n2_rand_dataset = rsatoolbox.data.TemporalDataset(n2_data_rand, descriptors = {'subj': ppnr}, channel_descriptors={'channels': true_channels_rand}, obs_descriptors={'conds': concat_stim_rand_n2}, time_descriptors={'time': t})
    n3_rand_dataset = rsatoolbox.data.TemporalDataset(n3_data_rand, descriptors = {'subj': ppnr}, channel_descriptors={'channels': true_channels_rand}, obs_descriptors={'conds': concat_stim_rand_n3}, time_descriptors={'time': t})

    n1_pred_dataset = rsatoolbox.data.TemporalDataset(n1_data_pred, descriptors = {'subj': ppnr}, channel_descriptors={'channels': true_channels_pred}, obs_descriptors={'conds': concat_stim_pred_n1}, time_descriptors={'time': t})
    n2_pred_dataset = rsatoolbox.data.TemporalDataset(n2_data_pred, descriptors = {'subj': ppnr}, channel_descriptors={'channels': true_channels_pred}, obs_descriptors={'conds': concat_stim_pred_n2}, time_descriptors={'time': t})
    n3_pred_dataset = rsatoolbox.data.TemporalDataset(n3_data_pred, descriptors = {'subj': ppnr}, channel_descriptors={'channels': true_channels_pred}, obs_descriptors={'conds': concat_stim_pred_n3}, time_descriptors={'time': t})
    #form the 6 datasets/conditions: block type (random/predictable) * position (n-1/n-2/n-3)


    #COVARIANCE MATRICES
    for k in range(1,4):
        split_data = vars()[f'n{k}_rand_dataset'].split_time('time')
        time_new = vars()[f'n{k}_rand_dataset'].time_descriptors['time']
        tp_for_covmat = []
        for tp in split_data:
            tp_single = tp.convert_to_dataset('time')
            tp_for_covmat.append(rsatoolbox.data.noise.cov_from_unbalanced(tp_single, obs_desc='conds', dof=None, method='shrinkage_diag'))
        tp_for_covmat = np.stack(tp_for_covmat)
        vars()[f'cov_mat_rand_n{k}']=np.mean(tp_for_covmat, axis=0)
        
        
        split_data = vars()[f'n{k}_pred_dataset'].split_time('time')
        time_new = vars()[f'n{k}_pred_dataset'].time_descriptors['time']
        tp_for_covmat = []
        for tp in split_data:
            tp_single = tp.convert_to_dataset('time')
            tp_for_covmat.append(rsatoolbox.data.noise.cov_from_unbalanced(tp_single, obs_desc='conds', dof=None, method='shrinkage_diag'))
        tp_for_covmat = np.stack(tp_for_covmat)
        vars()[f'cov_mat_pred_n{k}']=np.mean(tp_for_covmat, axis=0)
    #calculate covariance matrices per condition/timepoint, then average over timepoints


    #RSA
    for k in range(1,4):
        split_data = vars()[f'n{k}_rand_dataset'].split_time('time')
        time_new = vars()[f'n{k}_rand_dataset'].time_descriptors['time']

        rdms = []
        for dat in split_data:
            dat_single = dat.convert_to_dataset('time')
            rdms.append(calc_rdm_unbalanced(dat_single, method='crossnobis', descriptor='conds', noise=vars()[f'cov_mat_rand_n{k}'], cv_descriptor=None))
            
        rdms_data = concat(rdms)
        rdms_data.rdm_descriptors['time'] = time_new
        all_results.append(rdms_data)
        #calcuate representational dissimilarity matrices per condition/timepoint/participant in random datasets
        
        split_data = vars()[f'n{k}_pred_dataset'].split_time('time')
        time_new = vars()[f'n{k}_pred_dataset'].time_descriptors['time']

        rdms = []
        for dat in split_data:
            dat_single = dat.convert_to_dataset('time')
     
            rdms.append(calc_rdm_unbalanced(dat_single, method='crossnobis', descriptor='conds', noise=vars()[f'cov_mat_pred_n{k}'], cv_descriptor=None))

        rdms_data = concat(rdms)
        rdms_data.rdm_descriptors['time'] = time_new
        all_results.append(rdms_data)
        #calcuate representational dissimilarity matrices per condition/timepoint/participant in predictable datasets

        #element order of 'all_results' variable is n1_rand, n1_pred, n2_rand, n2_pred, n3_rand, n3_pred


#CHECK IF PATTERNS (ORDER OF STIMULUS PAIRS) ARE THE SAME WITHIN A TIMEPOINT (FOR RDM PLOTS)
patterns=[]
for row in all_results:
    for element in row:
        patterns.append(element.pattern_descriptors['conds'])
patterns=np.array(patterns)
patterns=np.reshape(patterns, (114,501,3))
        
for i in range(114):
    for j in range(500):
        if patterns[i,j,0] == patterns[i,j+1,0] and patterns[i,j,1] == patterns[i,j+1,1] and patterns[i,j,2] == patterns[i,j+1,2]:
            continue
        else:
            print(i,j)
            
#nothing printed. stimulus orders are the same within 114 (within each 501 timepoints, each participant/condition pair)


#CHECK IF PATTERNS (ORDER OF STIMULUS PAIRS) ARE THE SAME BETWEEN TIMEPOINTS (FOR RDM PLOTS)
patterns = []
for i in range(len(all_results)):
    patterns.append(all_results[i].pattern_descriptors['conds'])
patterns=np.array(patterns) 

#it's not the same between 114 results of each 501 timepoint
#I need to reorder dissimilarities accordingly before plotting


#FIX DISSIMILARITY PAIR ORDERS
all_dis = []
for i in range(len(all_results)):
    new_dis = all_results[i].get_vectors()
    all_dis.append(new_dis)
#get all dissimilarities as vectors

for i in range(len(all_dis)):
    all_dis[i] = np.mean(all_dis[i], axis=0)   
all_dis=np.array(all_dis)
#average vectors within each timepoint

ordered_dis=np.zeros((114,3))
for i in range(114):
    if patterns[i,0] == 'face' and patterns[i,1] == 'house' and patterns[i,2] == 'chair':
        ordered_dis[i] = (all_dis[i,0], all_dis[i,1], all_dis[i,2])          
    elif patterns[i,0] == 'face' and patterns[i,1] == 'chair' and patterns[i,2] == 'house':
        ordered_dis[i] = (all_dis[i,1], all_dis[i,0], all_dis[i,2])         
    elif patterns[i,0] == 'house' and patterns[i,1] == 'face' and patterns[i,2] == 'chair':
        ordered_dis[i] = (all_dis[i,0], all_dis[i,2], all_dis[i,1])         
    elif patterns[i,0] == 'house' and patterns[i,1] == 'chair' and patterns[i,2] == 'face':
        ordered_dis[i] = (all_dis[i,1], all_dis[i,2], all_dis[i,0])         
    elif patterns[i,0] == 'chair' and patterns[i,1] == 'face' and patterns[i,2] == 'house':
        ordered_dis[i] = (all_dis[i,2], all_dis[i,0], all_dis[i,1])        
    elif patterns[i,0] == 'chair' and patterns[i,1] == 'house' and patterns[i,2] == 'face':
        ordered_dis[i] = (all_dis[i,2], all_dis[i,1], all_dis[i,0])
#reorder dissimilarities to the pairs of (face-house/face-chair/house-chair)

n1_rand_rdm = ordered_dis[0::6]
n1_pred_rdm = ordered_dis[1::6]
n2_rand_rdm = ordered_dis[2::6]
n2_pred_rdm = ordered_dis[3::6]
n3_rand_rdm = ordered_dis[4::6]
n3_pred_rdm = ordered_dis[5::6] 
#divide vectors per condition per participant
  
def average_RDMs(rdm_list):
    sum_array = np.zeros((3), dtype=float)   
    for row in rdm_list:
        sum_array += row

    average_array = sum_array / len(rdm_list)   
    return average_array
    
n1_rand_avg_rdm = average_RDMs(n1_rand_rdm)   
n1_pred_avg_rdm = average_RDMs(n1_pred_rdm)   
n2_rand_avg_rdm = average_RDMs(n2_rand_rdm)   
n2_pred_avg_rdm = average_RDMs(n2_pred_rdm)   
n3_rand_avg_rdm = average_RDMs(n3_rand_rdm)   
n3_pred_avg_rdm = average_RDMs(n3_pred_rdm) 
#average within conditions (over participants)

def make_matrix(vector):
    empty_matrix = np.zeros((3,3))
    empty_matrix[1,0] = vector[0]
    empty_matrix[0,1] = vector[0]
    empty_matrix[2,0] = vector[1]
    empty_matrix[0,2] = vector[1]
    empty_matrix[2,1] = vector[2]
    empty_matrix[1,2] = vector[2]
    return empty_matrix

n1_rand_mat = make_matrix(n1_rand_avg_rdm)
n1_pred_mat = make_matrix(n1_pred_avg_rdm)
n2_rand_mat = make_matrix(n2_rand_avg_rdm)
n2_pred_mat = make_matrix(n2_pred_avg_rdm)
n3_rand_mat = make_matrix(n3_rand_avg_rdm)
n3_pred_mat = make_matrix(n3_pred_avg_rdm)
#transform dissimilarity vectors into matrices for plotting

categorical= np.array([[0, 1, 1],
                       [1, 0, 1],
                       [1, 1, 0]])
#the categorical dissimilarity matrix to be compared to the empirical data matrices


#PLOT AVERAGED RDMS
def plot_rdm(matrix, title):
    plt.imshow(matrix, cmap='gray', vmin=0, vmax=1.2)
    plt.colorbar(label='Dissimilarity')
    plt.title(title)
    plt.xticks(np.arange(3), labels=['Face', 'House', 'Chair'])
    plt.yticks(np.arange(3), labels=['Face', 'House', 'Chair'])
    plt.show()

matrices = [categorical-n1_rand_mat, categorical-n1_pred_mat, 
            categorical-n2_rand_mat, categorical-n2_pred_mat,
            categorical-n3_rand_mat, categorical-n3_pred_mat]

titles = ['Average RDM of Random N-1', 'Average RDM of Predictable N-1',
          'Average RDM of Random N-2', 'Average RDM of Predictable N-2',
          'Average RDM of Random N-3', 'Average RDM of Predictable N-3']

for matrix, title in zip(matrices, titles):
    plot_rdm(matrix, title)


#BLOCK TYPE AND PLACEMENT OF ALL CONCATENATED BLOCKS
both_block_types = np.array(['random', 'predictable'])
concat_block_types = np.tile(both_block_types, 57)

second_column = np.array(['n-1', 'n-2', 'n-3'])
second_column = np.repeat(second_column, 2)
second_column = np.tile(second_column, 19)

result_array = np.column_stack((concat_block_types, second_column))
#order is n1_rand, n1_pred, n2_rand, n2_pred, n3_rand, n3_pred


#DECODING ACCURACY
off_diagonal = []
for k in range(len(all_results)):
    off_diagonal.append(all_results[k].dissimilarities)
#save the off-diagonal values of the matrices

off_diag_mean = []
for j in range(len(off_diagonal)):
    off_diag_mean.append((np.mean(off_diagonal[j], axis=1)).tolist())
off_diag_mean = np.array(off_diag_mean)
#calculate the mean of off-diagonal values

unique, counts = np.unique(result_array, axis=0, return_counts=True)
indices_dict = {}
for i, row in enumerate(result_array):
    row_tuple = tuple(row)
    if row_tuple not in indices_dict:
        indices_dict[row_tuple] = []
    indices_dict[row_tuple].append(i)
#find RDM indices of unique block type/placement duos

for key, indices in indices_dict.items():
    values_to_plot = off_diag_mean[indices]

    plt.figure()
    x_axis = t
    plt.plot(x_axis, values_to_plot.T)

    plt.title(f'{key}')
    plt.xlabel('Index')
    plt.ylabel('Value')

plt.show()


#TWO-WAY REPEATED MEASURES ANOVA
pp_per_block = np.arange(1,20)
pp_per_block = np.repeat(pp_per_block, 6)

anovas=[]
for j in range(len(t)):
    dataframe_anova = pd.DataFrame({'Participant' : pp_per_block,
        'Block Type': result_array[:,0],
        'Position': result_array[:,1],
        'Results': off_diag_mean[:,j]})
    anovas.append(pg.rm_anova(dv='Results', within=['Block Type', 'Position'], subject='Participant', 
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


#PLOT SIGNIFICANCE AFTER FDR
def plot_significance(data, label):
    plt.plot(np.linspace(-0.1, 0.4, 501), data)
    plt.axhline(y=0.05, color='r', linestyle='--', label='Threshold (0.05)')
    plt.xlabel('Timepoints')
    plt.ylabel('P Value')
    plt.title(label)
    plt.legend()
    plt.show()

block_type_main = fdr[1][np.arange(0,1502,3)]
position_main = fdr[1][np.arange(1,1503,3)]
interaction = fdr[1][np.arange(2,1504,3)]

plot_significance(block_type_main, 'Block Type Main Effect')
plot_significance(position_main, 'Position Main Effect')
plot_significance(interaction, 'Interaction Effect')


#PARTICIPANT ACCURACY PLOTS (AVERAGED OVER TIMEPOINTS)
new_mean=np.mean(off_diag_mean, axis=1)

def plot_accuracy_of_pp(data, title):
    colors = ['yellow' if i in [1, 2, 3, 4] else 'green' if i in [16,17,18,19,20] else 'blue' for i in np.arange(1,20)]
    plt.bar(np.arange(1,20), new_mean[data], color=colors)
    plt.xticks(np.arange(1,20), participant_nr)
    plt.title(title)
    plt.show()

new_ind=np.arange(0,114,6)

accuracy_plot_data = [new_ind, new_ind+1, 
            new_ind+2, new_ind+3,
            new_ind+4, new_ind+5]

titles = ['Random N-1', 'Predictable N-1',
          'Random N-2', 'Predictable N-2',
          'Random N-3', 'Predictable N-3']

for data, title in zip(accuracy_plot_data, titles):
    plot_accuracy_of_pp(data, title)


#TIMEPOINT ACCURACY PLOTS (AVERAGED OVER PARTICIPANTS)
def sem_tp(pp_tp_data):
    sem = []
    for i in range(501):
        sem.append((np.std(pp_tp_data[:,i]) / np.sqrt(len(pp_tp_data[:,i]))))
    return sem
#calculate SEM per timepoint

def plot_tp_acc(data, title):
    new_mean=np.mean(off_diag_mean[data,:], axis=0)
    new_sem = sem_tp(off_diag_mean[data,:])

    plt.figure(figsize=(10, 6))
    plt.fill_between(t, (new_mean - new_sem), (new_mean + new_sem), alpha=0.2, color='blue')
    plt.plot(t, new_mean, color='black')
    plt.ylim(-0.75, 1.55)
    plt.title(title)
    plt.ylabel('Decoding magnitude (a.u.)')
    plt.xlabel('Time (s)')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.axvline(x=0, color='r', linestyle='--')
    plt.show()
    
new_ind=np.arange(0,114,6)

accuracy_plot_data = [new_ind, new_ind+1, 
            new_ind+2, new_ind+3,
            new_ind+4, new_ind+5]

titles = ['Random N-1', 'Predictable N-1',
          'Random N-2', 'Predictable N-2',
          'Random N-3', 'Predictable N-3']

for data, title in zip(accuracy_plot_data, titles):
    plot_tp_acc(data, title)
    

#ACCURACY BARPLOT (AVERAGED OVER TIMEPOINTS AND PARTICIPANTS)
cond_sem = []
#standard error of the mean per dataset
cond_mean = []
#mean per dataset
cond_all_res = []
#results per dataset

new_mean = np.mean(off_diag_mean, axis=1)
new_ind=np.arange(0,114,6)

def sem_and_mean(indices):
    cond_all = new_mean[indices]
    cond_all_res.append(cond_all)
    cond_mean.append(np.mean(cond_all))
    sample_std = np.std(cond_all, ddof=1)
    sample_size = len(cond_all)
    cond_sem.append(sample_std / np.sqrt(sample_size))
    
barplot_ind = [new_ind, new_ind+1, 
            new_ind+2, new_ind+3,
            new_ind+4, new_ind+5]  

for ind_list in barplot_ind:
    sem_and_mean(ind_list)
    
titles = ['Random N-1', 'Predictable N-1',
          'Random N-2', 'Predictable N-2',
          'Random N-3', 'Predictable N-3']    

plt.bar(titles, cond_mean, yerr=cond_sem, align='center', alpha=0.7, capsize=10)
plt.axhline(y=0, color='r', linestyle='--')
plt.xticks(rotation=25)
plt.ylabel('Decoding magnitude (a.u.)')
plt.show()


#PAIRED SAMPLE T-TESTS
ttest_n1 = stats.ttest_rel(cond_all_res[0], cond_all_res[1])
#compare 'Random N-1' and 'Predictable N-1'
ttest_n2 = stats.ttest_rel(cond_all_res[2], cond_all_res[3])
#compare 'Random N-2' and 'Predictable N-2'
ttest_n3 = stats.ttest_rel(cond_all_res[4], cond_all_res[5])
#compare 'Random N-3' and 'Predictable N-3'

print(ttest_n1)
print(ttest_n2)
print(ttest_n3)
 

#T-TEST FOR PREDICTABLE N-3
p3_t_stat, p3_p_value = stats.ttest_1samp(cond_all_res[5], 0, alternative='greater')
#compare 'Predictable N-3' to 0 mean
print(p3_t_stat , p3_p_value)


#FREQUENCY OF ALL SELECTED CHANNELS
flat_list_chan_rand = [item for sublist in chan_rand for item in sublist]
flat_list_chan_pred = [item for sublist in chan_pred for item in sublist]

rand_frequency = {}
for item in flat_list_chan_rand:
   if item in rand_frequency:
      rand_frequency[item] += 1
   else:
      rand_frequency[item] = 1

pred_frequency = {}
for item in flat_list_chan_pred:
   if item in pred_frequency:
      pred_frequency[item] += 1
   else:
      pred_frequency[item] = 1

sorted_pred_frequency = dict(sorted(pred_frequency.items(), key=lambda x: x[1], reverse=True))
sorted_rand_frequency = dict(sorted(rand_frequency.items(), key=lambda x: x[1], reverse=True))

df_pred = pd.DataFrame(list(sorted_pred_frequency.items()), columns=["Channel", "Frequency"])
df_pred["Percentage"] = ((df_pred["Frequency"] * 100) / 95).round(2)
#generate tables of channel selection frequency with percentage in predictable blocks

df_rand = pd.DataFrame(list(sorted_rand_frequency.items()), columns=["Channel", "Frequency"])
df_rand["Percentage"] = ((df_rand["Frequency"] * 100) / 95).round(2)
#generate tables of channel selection frequency with percentage in random blocks