import glob
import h5py
import pandas as pd
import numpy as np
from scipy import misc
import time
import sys
import matplotlib
import os
import matplotlib.pyplot as plt
import random
import ROOT as rt
import sys

rt.gROOT.SetBatch(True)

features = ['EvtId','Energy', 'Px', 'Py', 'Pz', 'Pt', 'Eta', 'Phi', 
                    'vtxX', 'vtxY', 'vtxZ','ChPFIso', 'GammaPFIso', 'NeuPFIso',
                    'isChHad', 'isNeuHad', 'isGamma', 'isEle',  'isMu', 
                        'Charge'
           ]

def makeGridECAL(ECAL):

    HFN = [-5, -4.7, -4.525, -4.35, -4.175, -4, -3.825, -3.65, -3.475, -3.3, -3.125, -2.958]
    HFP = [2.958, 3.125, 3.3, 3.475, 3.65, 3.825, 4, 4.175, 4.35, 4.525, 4.7, 5]

    ### ECAL ###

    # assume 0.02 x 0.02 resolution in eta,phi in the barrel |eta| < 1.5
    for i in range(-85, 86):
        etaLowEdge = i*0.0174
        etaHighEdge = (i+1)*0.0174
        for k in range(-180,180):
            phiLowEdge = k * np.pi/180.0
            phiHighEdge = (k+1)*np.pi/180.0
            ECAL.AddBin(etaLowEdge,phiLowEdge,etaHighEdge,phiHighEdge)

    # assume 0.02 x 0.02 resolution in eta,phi in the endcaps 1.5 < |eta| < 3.0 (HGCAL- ECAL)
    for k in range(-180,180):
        phiLowEdge = k * np.pi/180.0
        phiHighEdge = (k+1)*np.pi/180.0
        for i in range(0,85):
            etaLowEdge = -2.958 + i * 0.0174
            etaHighEdge = -2.958 + (i+1) * 0.0174
            ECAL.AddBin(etaLowEdge,phiLowEdge,etaHighEdge,phiHighEdge)

            etaLowEdge = 1.4964 + i * 0.0174
            etaHighEdge = 1.4964 + (i+1) * 0.0174
            ECAL.AddBin(etaLowEdge,phiLowEdge,etaHighEdge,phiHighEdge)

    # 0.175 x (0.175 - 0.35) resolution in eta,phi in the HF 3.0 < |eta| < 5.0
    for k in range(-18,18):
        phiLowEdge = k * np.pi/18.0
        phiHighEdge = (k+1) * np.pi/18.0
        for i in range(len(HFN)-1):
            etaLowEdge = HFN[i]
            etaHighEdge = HFN[i+1]
            ECAL.AddBin(etaLowEdge,phiLowEdge,etaHighEdge,phiHighEdge)
        for i in range(len(HFP)-1):
            etaLowEdge = HFP[i]
            etaHighEdge = HFP[i+1]
            ECAL.AddBin(etaLowEdge,phiLowEdge,etaHighEdge,phiHighEdge)


    ### HCAL ###
def makeGridHCAL(HCAL):
    
    HFN = [-5, -4.7, -4.525, -4.35, -4.175, -4, -3.825, -3.65, -3.475, -3.3, -3.125, -2.958]
    HFP = [2.958, 3.125, 3.3, 3.475, 3.65, 3.825, 4, 4.175, 4.35, 4.525, 4.7, 5]

    # assume 0.087 x 0.087 resolution in eta,phi in the barrel |eta| < 1.5
    HBbins = [-1.566, -1.479, -1.392, -1.305, -1.218, -1.131, -1.044, -0.957, -0.87, -0.783, -0.696, 
              -0.609, -0.522, -0.435, -0.348, -0.261, -0.174, -0.087, 0, 0.087, 0.174, 0.261, 
              0.348, 0.435, 0.522, 0.609, 0.696, 0.783, 0.87, 0.957, 1.044, 1.131, 1.218, 1.305, 
              1.392, 1.479, 1.566, 1.65]

    for k in range(-36, 36):
        phiLowEdge = k * np.pi/36.0
        phiHighEdge = (k+1) * np.pi/36.0
        for i in range(len(HBbins)-1):
            etaLowEdge = HBbins[i]
            etaHighEdge = HBbins[i+1]
            HCAL.AddBin(etaLowEdge,phiLowEdge,etaHighEdge,phiHighEdge)

    # assume 0.02 x 0.02 resolution in eta,phi in the endcaps 1.5 < |eta| < 3.0 (HGCAL- HCAL)
    for k in range(-180,180):
        phiLowEdge = k * np.pi/180.0
        phiHighEdge = (k+1)*np.pi/180.0
        for i in range(85):
            etaLowEdge = -2.958 + i * 0.0174
            etaHighEdge = -2.958 + (i+1) * 0.0174
            HCAL.AddBin(etaLowEdge,phiLowEdge,etaHighEdge,phiHighEdge)
            etaLowEdge = 1.4964 + i * 0.0174
            etaHighEdge = 1.4964 + (i+1) * 0.0174
            HCAL.AddBin(etaLowEdge,phiLowEdge,etaHighEdge,phiHighEdge)

    # 0.175 x (0.175 - 0.35) resolution in eta,phi in the HF 3.0 < |eta| < 5.0
    for k in range(-18,18):
        phiLowEdge = k * np.pi/18.0
        phiHighEdge = (k+1) * np.pi/18.0
        for i in range(len(HFN)-1):
            etaLowEdge = HFN[i]
            etaHighEdge = HFN[i+1]
            HCAL.AddBin(etaLowEdge,phiLowEdge,etaHighEdge,phiHighEdge)
        for i in range(len(HFP)-1):
            etaLowEdge = HFP[i]
            etaHighEdge = HFP[i+1]
            HCAL.AddBin(etaLowEdge,phiLowEdge,etaHighEdge,phiHighEdge)
    #return ECAL, HCAL

def fillEvent(d,i,ECALGamma, ECALTrack,HCAL):
    data = d[int(i),...]

    makeGridECAL(ECALGamma)
    makeGridECAL(ECALTrack)
    makeGridHCAL(HCAL)

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
            HCAL.Fill(eta,phi,pT)
        elif ptype == 0 or ptype == 3 or ptype == 4: # Track
            ECALTrack.Fill(eta,phi,pT)
        else: # Gamma
            ECALGamma.Fill(eta,phi,pT) 
    return ECALGamma, ECALTrack, HCAL

        
def do_it_all( sample, index, label, isval=False , lowerLimit=None, upperLimit = None ):
    rt.gStyle.SetOptStat(0)
    rt.gStyle.SetPadBorderSize(0)
    rt.gStyle.SetTitleSize(0)
    rt.gStyle.SetOptTitle(0)
    rt.gStyle.SetPalette(rt.kGreyScale)

    start = time.mktime(time.gmtime())
    dataset = None
    N=10
    if not lowerLimit:
        lowerLimit = 0
        upperLimit = sample.shape[0]
    if upperLimit > sample.shape[0]:
        upperLimit = sample.shape[0]
    if upperLimit < lowerLimit:
        sys.exit("Lower limit is higher than upper limit.")
    for i in range(lowerLimit, upperLimit):
        if i%N==0: 
            now = time.mktime(time.gmtime())
            so_far = now-start
            print i, so_far,"[s]"
            if i:
                eta = (so_far/(i-lowerLimit)* (upperLimit-lowerLimit)) - so_far
                print "finishing in", int(eta),"[s]", int(eta/60.),"[m]"
        ECALGamma = rt.TH2Poly()
        ECALTrack = rt.TH2Poly()
        HCAL = rt.TH2Poly()
        ECALGamma.SetName("ECAL")
        ECALTrack.SetName("Tracker")
        HCAL.SetName("HCAL")
        ECALTrack.GetXaxis().SetLabelSize(0)
        ECALTrack.GetYaxis().SetLabelSize(0)
        ECALGamma.GetXaxis().SetLabelSize(0)
        ECALGamma.GetYaxis().SetLabelSize(0)
        HCAL.GetXaxis().SetLabelSize(0)
        HCAL.GetYaxis().SetLabelSize(0)
        ECALGamma, ECALTrack, HCAL = fillEvent(sample, i, ECALGamma, ECALTrack, HCAL)
        print ("Max ECAL, Track, HCAL = {},{},{}".format(ECALGamma.GetMaximum(), ECALTrack.GetMaximum(), HCAL.GetMaximum()))
        #ECALGamma.GetZaxis().SetRangeUser(0,500)
        #ECALTrack.GetZaxis().SetRangeUser(0,1000)
        #HCAL.GetZaxis().SetRangeUser(0,500)
        c1 = rt.TCanvas("c1","c1",1200,400)
        c1.SetTopMargin(0)
        c1.SetBottomMargin(0)
        c1.SetRightMargin(0)
        c1.SetLeftMargin(0)

        c1.Divide(3,1,0,0)
        c1.cd(1)
        rt.gPad.SetLogz()
        ECALGamma.Draw("COL")
        c1.cd(2)
        rt.gPad.SetLogz()
        ECALTrack.Draw("COL")
        c1.cd(3)
        rt.gPad.SetLogz()
        HCAL.Draw("COL")
        c1.Draw()
        event_type = "QCD"
        if label == 1: event_type = "TTbar"
        if label == 2: event_type = "WJets"
        savedest = '/eos/cms/store/group/dpg_hcal/comm_hcal/qnguyen/RawImage/'+event_type+'/'
        if isval: savedest = savedest.replace('train','val')
        if not os.path.isdir(savedest): os.makedirs(savedest)
        c1.SaveAs(savedest+index+'_'+str(i)+'.png')
        out = rt.TFile(savedest+index+'_'+str(i)+'.root','recreate')
        ECALGamma.Write()
        ECALTrack.Write()
        HCAL.Write()
        out.Close()

def nf( fn ):
    return     fn.rsplit('/',1)[0]+'/images/'+fn.rsplit('/',1)[-1]

def change_directory(fn):
    if "train" in fn:
        return "/eos/cms/store/group/phys_susy/razor/thong/Delphes/train/"+fn.rsplit('/',1)[-1]
    if "val" in fn:
        return "/eos/cms/store/group/phys_susy/razor/thong/Delphes/val/"+fn.rsplit('/',1)[-1]

def make_reduced( f ) :
    if type(f) == str:
        f = h5py.File(f)    
    pf = f['Particles']
    reduced = np.zeros( (pf.shape[0], 801, 4))
    reduced[...,0] = f['Particles'][...,features.index('Eta')] 
    reduced[...,1] = f['Particles'][...,features.index('Phi')] 
    #reduced[...,2] = f['Particles'][...,features.index('Pt')] 
    reduced[...,2] = f['Particles'][...,features.index('Pt')]
    reduced[...,3] = np.argmax( f['Particles'][..., 14:19], axis=-1)
    #f.close()
    return reduced

def convert_sample( fn, lowerLimit=None, upperLimit = None ):
    f = h5py.File(fn)    
    reduced = make_reduced(f)
    print "Converting",fn 
    index = fn.rsplit('/',1)[-1].replace('.h5','')
    isval = "val" in fn
    if "qcd" in fn: label = 0
    if "ttbar" in fn: label = 1
    if "wjets" in fn: label = 2
    ds = do_it_all( reduced,index, label , isval, lowerLimit, upperLimit)
    f.close()
    print "Converted"
    
if __name__ == "__main__":
    if len(sys.argv)>1:
        ## make a special file
        if len(sys.argv)>2:
            lowerLimit = int(sys.argv[2]) 
            upperLimit = int(sys.argv[3])
        else:
            lowerLimit = None
            upperLimit = None
        convert_sample(sys.argv[1], lowerLimit, upperLimit)
    else:
        fl = []
        fl.extend(glob.glob('/eos/cms/store/cmst3/group/dehep/TOPCLASS/REDUCED_IsoLep/*/*.h5'))
        #fl.extend(glob.glob('/bigdata/shared/Delphes/np_datasets_new/3_way/MaxLepDeltaR_des/val/*.h5'))
        random.shuffle( fl )
        every = 5
        N= None
        for i,fn in enumerate(fl):
            com = 'python TransformToRawImage.py %s'%( fn)
            if N: com += ' %d'%N
            wait = (i%every==(every-1))
            if not wait: com +='&'
            print com
            os.system(com)
            if wait and N:
                time.sleep( 60 )

