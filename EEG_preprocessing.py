###EEG PREPROCESSING###
#EEG data from simultaneous EEG-fMRI
#gradient artifact was removed via BrainVision Analyzer 2.2
#19 participants. 2 predictable & 2 random blocks per participant
#the first 3 participants (ID 1, 2, 6) have 2 predictable & 1 random blocks
#there are 4 visual stimulus types in all blocks: face, house, chair, impulse


#IMPORT
import os
import mne
import mne.io
import mne.preprocessing
from mne.preprocessing import ICA, create_eog_epochs
import numpy as np
import pandas as pd
import statistics
import glob


#PARTICIPANT IDs, BLOCK NUMBERS AND DIRECTORIES
data_dir='/Users/yagmu/Desktop/NewEEG'
participant_nr=list(range(1,29))
remove_pp=[3,4,5,7,10,11,12,21,27]
for na_pp in range(len(remove_pp)):
    participant_nr.remove(remove_pp[na_pp])
#all participant IDs (19) after removing missing numbers
    
no_ecg_pp=list(range(13,21))
no_ecg_pp.insert(0,9)
no_ecg_pp.append(22)
#10 participant IDs with 32 channels (2 channels are duplicated, 30 channels will be used), and without ECG/EOG channel

full_32_pp=list(range(23,29))
del full_32_pp[-2]
#5 participant IDs with 32 channels, ECG channel but no EOG channel

#the remaining 4 participants have 64 channels (28 channels are usable), and no ECG/EOG channel 

for ppnr in participant_nr:
    if ppnr==1:
        block_nr=[2,3,4]
    elif ppnr==2 or ppnr==6:
        block_nr=[1,2,3]
    else:
        block_nr=list(range(1,5))
    #first 3 participants (ID 1, 2, 6) don't have data from all 4 block but from only 3       

    for bnr in block_nr:
        if ppnr<10:
            beh_dir=glob.glob(f'/Users/yagmu/Desktop/Analysis/pp0{ppnr}*')
            vhdr_dir=data_dir + '/pp0' + str(ppnr) + '_run' + str(bnr) + '_GA_CBA.vhdr'
            ppnr_folder=data_dir + '/plots/pp0' + str(ppnr)
            
        else:
            beh_dir=glob.glob(f'/Users/yagmu/Desktop/Analysis/pp{ppnr}*')
            vhdr_dir=data_dir + '/pp' + str(ppnr) + '_run' + str(bnr) + '_GA_CBA.vhdr'
            ppnr_folder=data_dir + '/plots/pp' + str(ppnr)

        bnr_folder = os.path.join(ppnr_folder, 'run' + str(bnr))
        os.makedirs(ppnr_folder, exist_ok=True)
        os.makedirs(bnr_folder, exist_ok=True)
        #create folders to save plots
        
        
        #READ RAW DATA
        os.chdir(data_dir)
        raw_run=mne.io.read_raw_brainvision(vhdr_dir, preload=True)
        ch_nr=len(raw_run.ch_names)
        
        
        #READ BEHAVIORAL DATA TO EXTRACT STIMULUS LABELS AND TIMING
        excel_files = glob.glob(f'{beh_dir[0]}/ID_{ppnr}_Block_{bnr}_Seq*.xlsx')
        beh = pd.read_excel(excel_files[0], header=None)
        mri_start=beh.iloc[2,1]
        #read the excel file and save when mri machine is on
        
        real_mristart=[]
        for j in range(len(raw_run._annotations.description)):
            if raw_run._annotations.description[j] == 'Response/R128':
                real_mristart=raw_run._annotations.onset[j]
                break
        #the first 'Response/R128' annotation marks the first mri scan time
        
        sequence_items = []
        for col in beh.columns[1:]:
            cell_value = beh.iloc[7][col]
            if isinstance(cell_value, str) and cell_value.strip():
                sequence_items.append(cell_value)
        #save stimulus labels from excel

        cells = []
        for col in beh.columns[1:]:
            cell_value = beh.iloc[8][col]
            if isinstance(cell_value, (int, float)):
                cells.append(cell_value)
        stimulus_times = [x - mri_start + real_mristart for x in cells]
        #save stimulus timings from excel. clock synchronize with mri start
        
        
        #ANNOTATE THE STIMULI & CROP
        my_annot=mne.Annotations(onset=stimulus_times, duration=0.032, description=sequence_items, orig_time=raw_run.info['meas_date'])
        raw_run.set_annotations(my_annot)
        #annotate labels

        break_annots = mne.preprocessing.annotate_break(raw=raw_run, min_break_duration=1, t_start_after_previous=0.7, t_stop_before_next=0.2) 
        raw_run.crop(tmin=break_annots.duration[0], tmax=break_annots.onset[-1])
        #crop the data to have only the experiment time
    
        figure=raw_run.plot(start=50., n_channels=ch_nr, scalings={"eeg":1e-4}, show_scrollbars=False);
        plot_path = os.path.join(bnr_folder, '0_GA_pp' + str(ppnr) + '_run' + str(bnr) + '.png')
        figure.savefig(plot_path)
        #plot and save
        
        
        #(MANUAL) CBA REMOVAL 
        filt_hb = raw_run.copy().filter(l_freq=1.0, h_freq=None)
        #highpass filtering before cardioballistic artifact removal
        ica_hb = ICA(n_components=20, max_iter="auto", random_state=42)
        #generate 20 components
        ica_hb.fit(filt_hb)
        #fit the components to the filtered data
        
        figure=ica_hb.plot_sources(raw_run, title='ICA Component Sources for CBA without ECG', show_scrollbars=False)
        plot_path = os.path.join(bnr_folder, 'CBA-ICA_sources_no_ecg_pp' + str(ppnr) + '_run' + str(bnr) + '.png')
        figure.savefig(plot_path)
        #plot ICA sources and save

        ICA_for_hb = []
        num_inputs = int(input("How many ICA components (CBA) do you want to use? "))
        for i in range(num_inputs):
            component = int(input(f"Enter ICA component (CBA) to use for participant {ppnr} block {bnr} ({i+1}/{num_inputs}): "))
            ICA_for_hb.append(component)
        ica_hb.exclude = ICA_for_hb
        #to manually select which components fit a heartbeat pattern, see the ICA sources plot
        #answer on the console how many components to exclude from the data
        #afterwards enter the ID number (without ICA00 at the beginning) of component on the console when it is asked
        #it will ask to enter the ID numbers one by one (one prompt per component)
        
        run_hb=raw_run.copy()
        ica_hb.apply(run_hb)
        #apply the manual selection to the data
        
        figure=run_hb.plot(start=50., n_channels=ch_nr, scalings={"eeg":1e-4}, show_scrollbars=False);
        plot_path = os.path.join(bnr_folder, '1_GA_CBA_pp' + str(ppnr) + '_run' + str(bnr) + '.png')
        figure.savefig(plot_path)
        #plot after CBA removal and save
        

        #BANDPASS FILTERING
        iir_params = dict(order=1, ftype='butter')  
        iir_params = mne.filter.construct_iir_filter(iir_params, f_pass=[.1, 48], sfreq=1000, btype='bandpass', return_copy=False)  
        #set butterworth zero-phase (two-pass forward and reverse) non-causal bandpass filtering with order 4, threshold 0.1-48 μV
       
        if ppnr in no_ecg_pp or full_32_pp:
            run_bp = run_hb.copy().filter(l_freq=.1, h_freq=48, picks=['eeg'], method='iir', iir_params=iir_params, skip_by_annotation='BAD') 
        else:
            run_bp = run_hb.copy().filter(l_freq=.1, h_freq=48, picks=['eeg', 'eog'], method='iir', iir_params=iir_params, skip_by_annotation='BAD') 
        #apply the filter

        figure=run_bp.plot(start=50., n_channels=ch_nr, scalings={"eeg":1e-4}, show_scrollbars=False);
        plot_path = os.path.join(bnr_folder, '2_GA_CBA_BP_pp' + str(ppnr) + '_run' + str(bnr) + '.png')
        figure.savefig(plot_path) 
        #plot after filtering and save
        
    
        #FIND BAD CHANNELS
        if ppnr in no_ecg_pp: 
            first_temp=run_bp.copy()
            temp_nr=ch_nr
        elif ppnr in full_32_pp:
            first_temp=run_bp.copy().drop_channels(['ECG'])
            temp_nr=ch_nr-1
        else:
            first_temp=run_bp.copy().drop_channels(['EOG', 'ECG'])
            temp_nr=ch_nr-2
        #drop non-EEG channels and set channel number based on participant ID
        
        temp=first_temp.get_data(reject_by_annotation='omit')
        temp_std=np.std(temp, axis=1)
        #standard deviation of each channel
        std_std=np.std(temp_std)*2
        #standard deviation of each channel over time
        median=statistics.median(temp_std)
        
        for k in range (temp_nr):
            if temp_std[k]>median+std_std or temp_std[k]<median-std_std:
                name=first_temp.ch_names[k]
                run_bp.info['bads'].append(name)
        #bad channel if its standard deviation exceeds the range of the median standard deviation +/- 2*the second level standard deviation

        figure=run_bp.plot(start=50., n_channels=ch_nr, scalings={"eeg":1e-4}, show_scrollbars=False);
        plot_path = os.path.join(bnr_folder, '3_GA_CBA_BP_badchn_pp' + str(ppnr) + '_run' + str(bnr) + '.png')
        figure.savefig(plot_path)
        #plot after excluding bad channels and save

    
        #EYEBLINK REMOVAL
        if 'Fp1' in run_bp.info['bads'] and 'Fp2' not in run_bp.info['bads']:
            eog_evoked = create_eog_epochs(run_bp, ch_name='Fp2', event_id=27, reject_by_annotation=True).average()
            eog_evoked.apply_baseline(baseline=(-0.2, 0))
            figure=eog_evoked.plot_joint(title='Eyeblink Evoked from Fp2', exclude='bads')
            plot_path = os.path.join(bnr_folder, 'Eyeblink_evoked_Fp2_pp' + str(ppnr) + '_run' + str(bnr) + '.png')
            figure.savefig(plot_path)

            filt_eye = run_bp.copy().filter(l_freq=1.0, h_freq=None)
            ica_eye = ICA(n_components=20, max_iter="auto", random_state=97)
            ica_eye.fit(filt_eye)
            eog_indices, eog_scores=ica_eye.find_bads_eog(run_bp, ch_name='Fp2')
        
        elif 'Fp1' in run_bp.info['bads'] and 'Fp2' in run_bp.info['bads']:
            eog_evoked = create_eog_epochs(run_bp, ch_name='F7', event_id=27, reject_by_annotation=True).average()
            eog_evoked.apply_baseline(baseline=(-0.2, 0))
            figure=eog_evoked.plot_joint(title='Eyeblink Evoked from F7', exclude='bads')
            plot_path = os.path.join(bnr_folder, 'Eyeblink_evoked_F7_pp' + str(ppnr) + '_run' + str(bnr) + '.png')
            figure.savefig(plot_path)

            filt_eye = run_bp.copy().filter(l_freq=1.0, h_freq=None)
            ica_eye = ICA(n_components=20, max_iter="auto", random_state=97)
            ica_eye.fit(filt_eye)
            eog_indices, eog_scores=ica_eye.find_bads_eog(run_bp, ch_name='F7')
            
        else:
            eog_evoked = create_eog_epochs(run_bp, ch_name='Fp1', event_id=27, reject_by_annotation=True).average()
            eog_evoked.apply_baseline(baseline=(-0.2, 0))
            figure=eog_evoked.plot_joint(title='Eyeblink Evoked from Fp1', exclude='bads')
            plot_path = os.path.join(bnr_folder, 'Eyeblink_evoked_Fp1_pp' + str(ppnr) + '_run' + str(bnr) + '.png')
            figure.savefig(plot_path)
    
            filt_eye = run_bp.copy().filter(l_freq=1.0, h_freq=None)
            ica_eye = ICA(n_components=20, max_iter="auto", random_state=97)
            ica_eye.fit(filt_eye)
            eog_indices, eog_scores=ica_eye.find_bads_eog(run_bp, ch_name='Fp1')
        #use Fp1, Fp2, or F7 (in this order of preference) to compare eyeblink ICAs with a frontal channel
        #plot the eyeblink template and save
            
        ica_eye.exclude = eog_indices
        figure=ica_eye.plot_components(title='ICA Components for Eyeblink')
        plot_path = os.path.join(bnr_folder, 'ICA_components_Eyeblink_pp' + str(ppnr) + '_run' + str(bnr) + '.png')
        figure.savefig(plot_path)
        #plot scalp distributions of the components and save
        
        figure=ica_eye.plot_scores(eog_scores, title='ICA Component Scores for Eyeblink')
        plot_path = os.path.join(bnr_folder, 'ICA_scores_Eyeblink_pp' + str(ppnr) + '_run' + str(bnr) + '.png')
        figure.savefig(plot_path)
        #plot the ICA component scores (winning one(s) are shown red) and save
        
        run_clean=run_bp.copy()
        ica_eye.apply(run_clean)
        #apply the component selection
    
        figure=run_clean.plot(start=50., n_channels=ch_nr, scalings={"eeg":1e-4}, show_scrollbars=False);
        plot_path = os.path.join(bnr_folder, '4_GA_CBA_BP_badchn_eog_pp' + str(ppnr) + '_run' + str(bnr) + '.png')
        figure.savefig(plot_path)
        #plot after eyeblink removal and save


        #INTERPOLATE BAD CHANNELS
        run_intpl = run_clean.copy().interpolate_bads()
        figure=run_intpl.plot(start=50., n_channels=ch_nr, scalings={"eeg":1e-4}, show_scrollbars=False);
        plot_path = os.path.join(bnr_folder, '5_GA_CBA_BP_badchn_eog_intpl_pp' + str(ppnr) + '_run' + str(bnr) + '.png')
        figure.savefig(plot_path)
        #plot after interpolating bad channels and save


        #RE-REFERENCING
        run_intpl.set_eeg_reference('average')
        figure=run_intpl.plot(start=50., n_channels=ch_nr, scalings={"eeg":1e-4}, show_scrollbars=False);
        plot_path = os.path.join(bnr_folder, '6_GA_CBA_BP_badchn_eog_intpl_reref_pp' + str(ppnr) + '_run' + str(bnr) + '.png')
        figure.savefig(plot_path)
        #plot after re-referencing and save


        #EPOCHING
        events, event_id = mne.events_from_annotations(run_intpl)
        run_epochs = mne.Epochs(run_intpl, events, event_id, tmin=-0.1, tmax=0.4, baseline=None, preload=True, proj=False, reject_by_annotation=True)  
        #divide epochs starting from -0.1 relative to the stimulus presentation and 0.4 after (0.5 s, 501 timepoints)

        figure=run_epochs.plot(scalings={"eeg":1e-4}, n_epochs=20, n_channels=ch_nr, show_scrollbars=False);
        plot_path = os.path.join(bnr_folder, '7_GA_CBA_BP_badchn_eog_intpl_reref_epoched_pp' + str(ppnr) + '_run' + str(bnr) + '.png')
        figure.savefig(plot_path)
        #plot after dividing into epochs and save


        #BASELINE CORRECTION
        baseline = (-0.1, 0)
        run_epochs.apply_baseline(baseline)

        figure=run_epochs.plot(scalings={"eeg":1e-4}, n_channels=ch_nr, show_scrollbars=False);
        plot_path = os.path.join(bnr_folder, '8_GA_CBA_BP_badchn_eog_intpl_reref_epoched_basecor_pp' + str(ppnr) + '_run' + str(bnr) + '.png')
        figure.savefig(plot_path)
        #plot after baseline correction and save
    
    
        #BAD TRIALS
        temp_epoch = run_epochs.get_data(picks='eeg')
        std_per_epoch = np.std(temp_epoch, axis=2)
        #standard deviation of each epoch in each channel separately over timepoints
        mean_std_epoch = np.mean(std_per_epoch, axis=1)
        #mean standard deviations over channels (one value per epoch)

        median_epoch=statistics.median(mean_std_epoch)
        std_std_epoch=np.std(mean_std_epoch)*2
        bad_epoch_index=[]

        for l in range (1125):
            if mean_std_epoch[l]>median_epoch+std_std_epoch or mean_std_epoch[l]<median_epoch-std_std_epoch:
                bad_epoch_index.append(l)

        run_epochs_bad_trials=run_epochs.drop(bad_epoch_index)
        #epoch dropped if mean SD exceeds the range of the median of the mean standard deviations +/- 2*the standard deviation of the mean standard deviations


        #TRIALS WITH UPSIDE DOWN PICTURES
        image_ort=beh.iloc[3][1:]
        image_ort=image_ort.tolist()
        image_down=[]
        for i in range (1125):
            if run_epochs_bad_trials._annotations[i]=='burst':
                if image_ort[i]==1 or image_ort[i-1]==1 or image_ort[i-2]==1 or image_ort[i-3]==1:
                    image_down.append(i, i-1, i-2, i-3)
        #find the indices of upside down images in the impulse(burst) images or the 3 preceding images

        run_epochs_final=run_epochs_bad_trials.drop(image_down)
        #drop those 4 epochs if any of them are upside down
        
        figure=run_epochs_bad_trials.plot(scalings={"eeg":1e-4}, n_channels=ch_nr, show_scrollbars=False);
        plot_path = os.path.join(bnr_folder, '9_GA_CBA_BP_badchn_eog_intpl_reref_epoched_basecor_badtr&upsidedrop_pp' + str(ppnr) + '_run' + str(bnr) + '.png')
        figure.savefig(plot_path)
        #plot after dropping bad trials & upside down images and save
      
    
        #AVERAGE ERPs
        evoked_face = run_epochs_final['face'].average()
        figure=evoked_face.plot();
        plot_path = os.path.join(bnr_folder, 'evoked1_face_pp' + str(ppnr) + '_run' + str(bnr) + '.png')
        figure.savefig(plot_path)

        evoked_house = run_epochs_final['house'].average()
        figure=evoked_house.plot();
        plot_path = os.path.join(bnr_folder, 'evoked2_house_pp' + str(ppnr) + '_run' + str(bnr) + '.png')
        figure.savefig(plot_path)

        evoked_chair = run_epochs_final['chair'].average()
        figure=evoked_chair.plot();
        plot_path = os.path.join(bnr_folder, 'evoked3_chair_pp' + str(ppnr) + '_run' + str(bnr) + '.png')
        figure.savefig(plot_path)

        evoked_impulse = run_epochs_final['burst'].average()
        figure=evoked_impulse.plot();
        plot_path = os.path.join(bnr_folder, 'evoked4_impulse_pp' + str(ppnr) + '_run' + str(bnr) + '.png')
        figure.savefig(plot_path)
        #average the data of same condition epochs
        #plot average ERPs and save
        
    
        #SAVE EPOCH DATA
        file_name='pp' + str(ppnr) + '_run' + str(bnr) + '_preprocessed_epoched-epo.fif'
        run_epochs_final.save(file_name, overwrite=True)