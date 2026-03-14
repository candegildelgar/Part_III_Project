import numpy as np
import glob
import shutil
import obspy
import os
direc= ['/raid2/cg812/Checking_error/Gauss_2.0', '/raid2/cg812/Checking_error/Gauss_3.0', '/raid2/cg812/Checking_error/Gauss_4.0', '/raid2/cg812/Checking_error/Gauss_5.0', '/raid2/cg812/Checking_error/Gauss_6.0']
for gauss in direc:
    print(gauss)
    for sta in glob.glob(gauss +'/*'):
        for bin in glob.glob(sta + '/*'):
            print(bin)
            traces= np.array(glob.glob(bin +'/*[!.png]'))
            if len(traces) == 1 or len(traces)==2:
                filtered_traces = traces
            else:
                keep= np.zeros(len(traces), dtype=bool)
                for i in range(len(traces)):
                    for j in range(i+1, len(traces)):
                        st=obspy.read(traces[i])[0].data
                        comparison= obspy.read(traces[j])[0].data
                        try:
                            corr= np.corrcoef(st, comparison)[0, 1]
                        except:
                            print('differenet length traces')
                        if corr>=0.6:
                            keep[i]= True
                            keep[j]= True
                filtered_traces=traces[keep]
            for tr in filtered_traces:
                savepath=os.path.join('Refined_automatically', os.path.basename(gauss) + '/'+ os.path.basename(sta) + '/' + os.path.basename(bin))
                if not os.path.exists(savepath):
                        os.makedirs(savepath)
               
                shutil.copyfile(tr, savepath + '/' + os.path.basename(tr))
                shutil.copyfile(tr + '.png', savepath + '/' + os.path.basename(tr) + '.png')
                print('saving to' + savepath)
                
