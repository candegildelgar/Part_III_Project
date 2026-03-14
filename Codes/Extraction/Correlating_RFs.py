import pandas as pd
from textual.app import App
from textual_pandas.widgets import DataFrameTable
import os
cross_correlation= pd.read_csv('/raid2/cg812/Processed_RF/HRUR/cross_correlation_matrix')
cross_correlation=cross_correlation.drop(columns= ['2024-07-11T15:18:07.300000']
                                             , index=[3])                                


#savepath= os.path.join('/raid2/cg812/Processed_RF/HOTR/cross_correlation_matrix_final.csv')
#cross_correlation.to_csv(savepath)


    
class PandasApp(App):
    def compose(self):
        yield DataFrameTable()
    def on_mount(self):
        table = self.query_one(DataFrameTable)
        table.add_df(cross_correlation)
if __name__ == "__main__":
    app = PandasApp()
    app.run()

import obspy
import pandas as pd 
import numpy as np
import glob
rfstack= list()
event= list()
direc= glob.glob('/raid2/cg812/Processed_RF_different_method_for_refining_Gauss_4.5/VITI/bin_range340.0to350.0/*[!.png]')
for dat in direc:
        st= obspy.read(dat)
        rfstack.append(st[0].data)
        event.append(str(st[0].stats.starttime))


cross_correlation_matrix= pd.DataFrame(data=np.corrcoef(rfstack), columns= event)
cross_correlation_matrix