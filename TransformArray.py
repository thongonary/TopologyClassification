import glob
import h5py
import pandas as pd
import numpy as np
from scipy import misc
import time
import sys
import matplotlib
from skimage import draw 
import ROOT as rt
import os
import matplotlib.pyplot as plt
import random
from array import array
from root_numpy import hist2array

features = ['Energy', 'Px', 'Py', 'Pz', 'Pt', 'Eta', 'Phi', 
                    'vtxX', 'vtxY', 'vtxZ','ChPFIso', 'GammaPFIso', 'NeuPFIso',
                    'isChHad', 'isNeuHad', 'isGamma', 'isEle',  'isMu', 
                        #'Charge'
           ]

def makeGridEBEE():

    # assume 0.02 x 0.02 resolution in eta,phi in the barrel |eta| < 1.5

    phi_ebee = []
    for k in range(-180,181):
        phi_ebee.append(k*np.pi/180.0)

    eta_barrel = []
    for i in range(-85, 86):
        eta_barrel.append(i*0.0174)

    # assume 0.02 x 0.02 resolution in eta,phi in the endcaps 1.5 < |eta| < 3.0 (HGCAL- ECAL)
    
    eta_endcap1 = []
    eta_endcap2 = []
    
    for i in range(1,85):
        eta_endcap1.append(-2.958 + i * 0.0174)
        eta_endcap2.append(1.4964 + i * 0.0174)
        
    eta_ebee = np.concatenate((eta_endcap1, eta_barrel, eta_endcap2))
    
    return eta_ebee, phi_ebee
    
def makeGridForwardN():
    
    eta_forward_1 = [-5, -4.7, -4.525, -4.35, -4.175, -4, -3.825, -3.65, -3.475, -3.3, -3.125, -2.958]
    eta_forward_2 = [2.958, 3.125, 3.3, 3.475, 3.65, 3.825, 4, 4.175, 4.35, 4.525, 4.7, 5]
    
    phi_forward = []
    for k in range (-18,19):
        phi_forward.append(k * np.pi/18.0)

    return eta_forward_1, phi_forward

def makeGridForwardP():
    eta_forward_1 = [-5, -4.7, -4.525, -4.35, -4.175, -4, -3.825, -3.65, -3.475, -3.3, -3.125, -2.958]
    eta_forward_2 = [2.958, 3.125, 3.3, 3.475, 3.65, 3.825, 4, 4.175, 4.35, 4.525, 4.7, 5]
    
    phi_forward = []
    for k in range (-18,19):
        phi_forward.append(k * np.pi/18.0)

    return eta_forward_2, phi_forward

    
def makeGridHB():
    
    # assume 0.087 x 0.087 resolution in eta,phi in the barrel |eta| < 1.5
    eta_HB = [-1.566, -1.479, -1.392, -1.305, -1.218, -1.131, -1.044, -0.957, -0.87, -0.783, -0.696, 
              -0.609, -0.522, -0.435, -0.348, -0.261, -0.174, -0.087, 0, 0.087, 0.174, 0.261, 
              0.348, 0.435, 0.522, 0.609, 0.696, 0.783, 0.87, 0.957, 1.044, 1.131, 1.218, 1.305, 
              1.392, 1.479, 1.566]

    phi_barrel = []
    for k in range(-36, 37):
        phi_barrel.append(k * np.pi/36.0)
        
    return eta_HB, phi_barrel

def makeGridHEN():
    phi_HE = []
    for k in range(-180,181):
        phi_HE.append(k * np.pi/180.0)
    eta_HE = []
    for i in range(1,85):
        eta_HE.append(-2.958 + i * 0.0174)
    
    return eta_HE, phi_HE

def makeGridHEP():
    phi_HE = []
    for k in range(-180,181):
        phi_HE.append(k * np.pi/180.0)
    eta_HE = []
    for i in range(1,85):
        eta_HE.append(1.4964 + i * 0.0174)
    
    return eta_HE, phi_HE   

def showSEvent(d,i,show=True):
    def check_increasing(L):
        return all(x<y for x, y in zip(L, L[1:]))

    data = d[int(i),...]
    
    all_hists = []
    
    eta_forwardN, phi_forwardN = makeGridForwardN()    
    eta_forwardP, phi_forwardP = makeGridForwardP()
    eta_ebee, phi_ebee = makeGridEBEE()
    
    eta_hen, phi_hen = makeGridHEN()
    eta_hb, phi_hb = makeGridHB()
    eta_hep, phi_hep = makeGridHEP()
    
    assert check_increasing(eta_forwardN)
    assert check_increasing(eta_forwardP)
    assert check_increasing(phi_forwardN)
    assert check_increasing(phi_forwardP)
    assert check_increasing(eta_ebee)
    assert check_increasing(phi_ebee)
    assert check_increasing(eta_hen)
    assert check_increasing(phi_hen)
    assert check_increasing(eta_hb)
    assert check_increasing(phi_hb)    
    assert check_increasing(eta_hep)
    assert check_increasing(phi_hep)

    ECAL_ForwardN = rt.TH2F("ECAL_ForwardN","",len(eta_forwardN)-1, array('d',eta_forwardN), 
                            len(phi_forwardN)-1, array('d',phi_forwardN))

    ECAL_ForwardP = rt.TH2F("ECAL_ForwardP","",len(eta_forwardP)-1, array('d',eta_forwardP), 
                            len(phi_forwardP)-1, array('d',phi_forwardP))
    
    ECAL_EBEE = rt.TH2F("ECAL_EBEE","",len(eta_ebee)-1, array('d',eta_ebee), 
                            len(phi_ebee)-1, array('d',phi_ebee))
    
    ECAL_Gamma_ForwardN = rt.TH2F("ECAL_Gamma_ForwardN","",len(eta_forwardN)-1, array('d',eta_forwardN), 
                            len(phi_forwardN)-1, array('d',phi_forwardN))

    ECAL_Gamma_ForwardP = rt.TH2F("ECAL_Gamma_ForwardP","",len(eta_forwardP)-1, array('d',eta_forwardP), 
                            len(phi_forwardP)-1, array('d',phi_forwardP))
    
    ECAL_Gamma_EBEE = rt.TH2F("ECAL_Gamma_EBEE","",len(eta_ebee)-1, array('d',eta_ebee), 
                            len(phi_ebee)-1, array('d',phi_ebee))  
    
    
    HCAL_ForwardN = rt.TH2F("HCAL_ForwardN","",len(eta_forwardN)-1, array('d',eta_forwardN), 
                            len(phi_forwardN)-1, array('d',phi_forwardN))
    HCAL_ForwardP = rt.TH2F("HCAL_ForwardP","",len(eta_forwardP)-1, array('d',eta_forwardP), 
                            len(phi_forwardP)-1, array('d',phi_forwardP))
    HCAL_HEN = rt.TH2F("HCAL_HEN","",len(eta_hen)-1, array('d',eta_hen), 
                            len(phi_hen)-1, array('d',phi_hen))
    HCAL_HB = rt.TH2F("HCAL_HB","",len(eta_hb)-1, array('d',eta_hb), 
                            len(phi_hb)-1, array('d',phi_hb))
    HCAL_HEP = rt.TH2F("HCAL_HEP","",len(eta_hep)-1, array('d',eta_hep), 
                            len(phi_hep)-1, array('d',phi_hep))
    
    for ip in range(data.shape[0]):
        p_data = data[ip,:]
        eta = p_data[0]
        phi = p_data[1]
        if eta==0 and phi==0: 
            #print ip
            continue
        #pT = p_data[2]
        #lpT = min(max(np.log(pT)/5.,0.001), 10)*res/2.
        pT = p_data[2]
        ptype = int(p_data[3])
        if ptype == 1: # NeuHad. Fill HCAL:
            if (abs(eta) < 1.5): #Fill HB
                HCAL_HB.Fill(eta, phi, pT)
            elif (abs(eta) < 3): #Fill HE
                if eta > 0: #Fill HEP
                    HCAL_HEP.Fill(eta, phi, pT)
                else: #Fill HEN
                    HCAL_HEN.Fill(eta, phi, pT)
            else: #Fill HF
                if eta > 0: #Fill forward P
                    HCAL_ForwardP.Fill(eta, phi, pT)
                else: #Fill forward N
                    HCAL_ForwardN.Fill(eta, phi, pT)
                    
        elif ptype == 0 or ptype == 3 or ptype == 4: # Track. Fill ECAL
            if (abs(eta) < 3): #Fill ebee
                ECAL_EBEE.Fill(eta, phi, pT)
            else: #Fill ECAL Forward
                if eta > 0: #Fill forward P
                    ECAL_ForwardP.Fill(eta, phi, pT)
                else: #Fill forward N
                    ECAL_ForwardN.Fill(eta, phi, pT)
            
        else: # Gamma
            if (abs(eta) < 3): #Fill ebee
                ECAL_Gamma_EBEE.Fill(eta, phi, pT)
            else: #Fill ECAL Forward
                if eta > 0: #Fill forward P
                    ECAL_Gamma_ForwardP.Fill(eta, phi, pT)
                else: #Fill forward N
                    ECAL_Gamma_ForwardN.Fill(eta, phi, pT)
                    
    # Convert all hists to numpy arrays and append to all_hists
    all_hists.append(hist2array(ECAL_ForwardN))
    all_hists.append(hist2array(ECAL_EBEE))
    all_hists.append(hist2array(ECAL_ForwardP))
    all_hists.append(hist2array(ECAL_Gamma_ForwardN))
    all_hists.append(hist2array(ECAL_Gamma_EBEE))
    all_hists.append(hist2array(ECAL_Gamma_ForwardP))
    all_hists.append(hist2array(HCAL_ForwardN))
    all_hists.append(hist2array(HCAL_HEN))
    all_hists.append(hist2array(HCAL_HB))
    all_hists.append(hist2array(HCAL_HEP))
    all_hists.append(hist2array(HCAL_ForwardP))
    
    return all_hists

def do_it_all( sample ,limit=None ):
    start = time.mktime(time.gmtime())
    dataset = []
    N=100
    max_I = limit if limit else sample.shape[0]
    for i in range(max_I):
        if i%N==0: 
            now = time.mktime(time.gmtime())
            so_far = now-start
            print (i, so_far,"[s]")
            if i:
                eta = (so_far/i* max_I) - so_far
                print ("finishing in", int(eta),"[s]", int(eta/60.),"[m]")
        all_hists = showSEvent(sample, i, show=False)
        
        # Return 11 numpy arrays corresponding to 11 histograms
        if len(dataset) == 0:
            for hist in range(len(all_hists)):
                dataset.append(np.zeros((max_I,)+all_hists[hist].shape))
        for hist in range(len(all_hists)):
            dataset[hist][i,...] = all_hists[hist]
    return dataset

def nf( fn ):
    return     fn.rsplit('/',1)[0]+'/images/'+fn.rsplit('/',1)[-1]

def move_to_thong(fn):
    if "train" in fn:
        return "/bigdata/shared/LCDJets_RawArray/train/"+fn.rsplit('/',1)[-1]
    if "val" in fn:
        return "/bigdata/shared/LCDJets_RawArray/val/"+fn.rsplit('/',1)[-1]

def make_reduced( f ) :
    if type(f) == str:
        f = h5py.File(f)    
    pf = f['Particles']
    reduced = np.zeros( (pf.shape[0], 801, 4))
    reduced[...,0] = f['Particles'][...,features.index('Eta')] 
    reduced[...,1] = f['Particles'][...,features.index('Phi')] 
    #reduced[...,2] = f['Particles'][...,features.index('Pt')] 
    reduced[...,2] = np.minimum(np.log(np.maximum(f['Particles'][...,features.index('Pt')], 1.001))/5., 10)
    reduced[...,3] = np.argmax( f['Particles'][..., 13:], axis=-1)

    h_reduced = np.zeros( (pf.shape[0], 1, 4))
    #h_reduced[...,0,2] = f['HLF'][..., 1] # MET
    h_reduced[...,0,2] = np.minimum(np.maximum(np.log(f['HLF'][..., 1])/5.,0.001), 10) # MET
    h_reduced[...,0,1] = f['HLF'][..., 2] # MET-phi
    h_reduced[...,0,3 ] = int(5) ## met type

    reduced = np.concatenate( (reduced, h_reduced), axis=1)

    return reduced

def convert_sample( fn, limit=None ):
    f = h5py.File(fn)    
    reduced = make_reduced(f)
    #new_fn = nf(fn)
    new_fn = move_to_thong(fn)
    if limit:
        print ("Converting",fn,"into",new_fn,("for %s events"%limit)) 
    ds = do_it_all( reduced ,limit)
    n_f = h5py.File( new_fn,'w')
    #n_f['data'] = reduced 
    #n_f['Images'] = ds
    #n_f['Labels'] = f['Labels'][:limit,...] if limit else f['Labels'][...]
    checkNaN = []
    for i in range(len(ds)):
        checkNaN.append(np.isnan(ds[i]).any())
    if True in checkNaN:
        print ("%s has NaN after conversion" %fn)
    else:
        tmp = f['Labels'][:limit,...] if limit else f['Labels'][...]
        n_f.create_dataset('ECAL_ForwardN', data = ds[0], dtype = np.float32)
        n_f.create_dataset('ECAL_EBEE', data = ds[1], dtype = np.float32)
        n_f.create_dataset('ECAL_ForwardP', data = ds[2], dtype = np.float32)
        n_f.create_dataset('ECAL_Gamma_ForwardN', data = ds[3], dtype = np.float32)
        n_f.create_dataset('ECAL_Gamma_EBEE', data = ds[4], dtype = np.float32)
        n_f.create_dataset('ECAL_Gamma_ForwardP', data = ds[5], dtype = np.float32)
        n_f.create_dataset('HCAL_ForwardN', data = ds[6], dtype = np.float32)
        n_f.create_dataset('HCAL_HEN', data = ds[7], dtype = np.float32)
        n_f.create_dataset('HCAL_HB', data = ds[8], dtype = np.float32)
        n_f.create_dataset('HCAL_HEP', data = ds[9], dtype = np.float32)
        n_f.create_dataset('HCAL_ForwardP', data = ds[10], dtype = np.float32)
        n_f.create_dataset('Labels', data = tmp, dtype = np.uint8)
    n_f.close()


    
if __name__ == "__main__":
    if len(sys.argv)>1:
        ## make a special file
        limit = int(sys.argv[2]) if len(sys.argv)>2 else None
        convert_sample(sys.argv[1], limit)
    else:
        fl = []
        fl.extend(glob.glob('/bigdata/shared/Delphes/np_datasets_new/3_way/MaxLepDeltaR_des/train/*.h5'))
        fl.extend(glob.glob('/bigdata/shared/Delphes/np_datasets_new/3_way/MaxLepDeltaR_des/val/*.h5'))
        random.shuffle( fl )
        every = 5
        N= None
        for i,fn in enumerate(fl):
            com = 'python3 TransformArray.py %s'%( fn)
            if N: com += ' %d'%N
            wait = (i%every==(every-1))
            if not wait: com +='&'
            print (com)
            os.system(com)
            if wait and N:
                time.sleep( 60 )

