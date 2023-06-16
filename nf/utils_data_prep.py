import sys, os
import pickle as pk
import numpy as np


def load_density_halo_data(
        ji,
        nside_d,
        nbatch,
        nfilter,
        ncnn,
        fields_all,
        z=0,
        min_val_Mtot_tosub = None,
        min_vals_fields_tosub = None,
        sig_val_Mtot_todiv = None,
        sig_vals_fields_todiv = None,
        sdir='/pscratch/sd/s/spandey/camels/3D_grid/split_IllustrisTNG'
    ):
    # load the density data
    field = 'Mtot_Nbody'
    root_out     = sdir + '/%s/CV/subvol_NPE'%field
    folder_out = '%s/%d'%(root_out,ji)
    savefname_density_full = '%s/density_subvol_m_res_%d_z=%s_nbatch_%d_nfilter_%d_ncnn_%d.pk'%(folder_out,nside_d,z,nbatch,nfilter,ncnn)
    df_Mtot_Nbody = (pk.load(open(savefname_density_full, 'rb'))['density_cic_pad'])
    df_d_all = np.log(df_Mtot_Nbody + 1e-5)
    if min_val_Mtot_tosub is None:
        min_val_Mtot_tosub = np.amin(df_d_all)
    df_d_all -= min_val_Mtot_tosub
    if sig_val_Mtot_todiv is None:
        sig_val_Mtot_todiv = 5*np.std(df_d_all)
    df_d_all /= sig_val_Mtot_todiv

    root_out     = sdir + '/%s/CV/subvol_NPE'%fields_all[0]
    folder_out = '%s/%d'%(root_out,ji)
    savefname_density_full = '%s/density_subvol_m_res_%d_z=%s_nbatch_%d_nfilter_%d_ncnn_%d.pk'%(folder_out,nside_d,z,nbatch,nfilter,0)
    df_field = (pk.load(open(savefname_density_full, 'rb'))['density_cic_pad'])
    df_pred_all = np.zeros((df_field.shape + (len(fields_all),)))
    if min_vals_fields_tosub is None:
        min_vals_fields_tosub = np.zeros(len(fields_all))
        save_min_vals_fields_tosub = True
    
    if sig_vals_fields_todiv is None:
        sig_vals_fields_todiv = np.zeros(len(fields_all))
        save_sig_vals_fields_todiv = True

    for jf, field in enumerate(fields_all):
        root_out     = sdir + '/%s/CV/subvol_NPE'%field
        folder_out = '%s/%d'%(root_out,ji)
        savefname_density_full = '%s/density_subvol_m_res_%d_z=%s_nbatch_%d_nfilter_%d_ncnn_%d.pk'%(folder_out,nside_d,z,nbatch,nfilter,0)
        df_field = (pk.load(open(savefname_density_full, 'rb'))['density_cic_pad'])
        df_pred_all[...,jf] = np.log(df_field + 1e-5)

        if save_min_vals_fields_tosub:
            df_field_min = np.amin(df_pred_all[...,jf])
            min_vals_fields_tosub[jf] = df_field_min
        else:
            df_field_min = min_vals_fields_tosub[jf]
        df_pred_all[...,jf] -= df_field_min

        if save_sig_vals_fields_todiv:
            df_field_sig = 5*np.std(df_pred_all[...,jf])
            sig_vals_fields_todiv[jf] = df_field_sig
        else:
            df_field_sig = sig_vals_fields_todiv[jf]
        df_pred_all[...,jf] /= df_field_sig
            
    return df_d_all, df_pred_all, min_val_Mtot_tosub, min_vals_fields_tosub, sig_val_Mtot_todiv, sig_vals_fields_todiv



def prep_density_halo_cats(
        df_d_all, df_pred_all, nsims=None, nstart=None
    ):
    if nstart is None:
        if nsims is None:
            nsims = df_d_all.shape[0]
        # We only need the first nsims
        df_d_all = df_d_all[:nsims, ...]
        df_pred_all = df_pred_all[:nsims, ...]
    else:
        nend = nstart + nsims
        df_d_all = df_d_all[nstart:nend, ...]
        df_pred_all = df_pred_all[nstart:nend, ...]
        print(nstart, nend)
    # Now we reshape the number of halos into 2D array of shape number of sub-sim, nvoxels (per sub-sim)
    # Note that the number of sub-sim = nb**3

    # final dict with all the required data to run the model
    return_dict = {}
    return_dict['df_d_all'] = np.array([df_d_all])
    return_dict['df_pred_all'] = np.array([df_pred_all])
    return return_dict


def prep_density_halo_cats_batched(
        df_d_all_inp,
        df_pred_all_inp,
        nsims=None,
        nbatches=1,
    ):
    df_d_all_out = []
    df_pred_all_out = []
    for jb in range(nbatches):
        nstart = jb * nsims
        nend = (jb + 1) * nsims
        df_d_all = df_d_all_inp[nstart:nend, ...]
        df_d_all_out.append(df_d_all)
        df_pred_all = df_pred_all_inp[nstart:nend, ...]
        df_pred_all_out.append(df_pred_all)
        print(nstart, nend)
    df_d_all_out = np.array(df_d_all_out)
    df_pred_all_out = np.array(df_pred_all_out)
    df_pred_all_out = np.reshape(df_pred_all_out, (nbatches, nsims, df_pred_all_out.shape[2]* df_pred_all_out.shape[3]* df_pred_all_out.shape[4], df_pred_all_out.shape[5]))
    # final dict with all the required data to run the model
    return_dict = {}
    return_dict['df_d_all'] = np.array(df_d_all_out)
    return_dict['df_pred_all'] = np.array(df_pred_all_out)
    return return_dict
