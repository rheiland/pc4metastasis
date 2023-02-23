from fury3D import CreateScene, CreateSnapshots
import pcDataLoader as pc
from pathlib import Path
import numpy as np
import pandas as pd
import sys

def my_custom_coloring_function(df_cells):
    Gene_RFP = np.array([df_cells['genes_0']])
    RFP = np.where(Gene_RFP == 1)
    GFP = np.where(Gene_RFP == 0)
    # Colors
    Colors = np.ones((len(df_cells),4)) # 4 channels RGBO
    # RFP+ cells - Red
    Colors[RFP[1],0] = 1
    Colors[RFP[1],1] = 0
    Colors[RFP[1],2] = 0
    # GFP+ cells - Green
    Colors[GFP[1],0]= 0
    Colors[GFP[1],1] = 1
    Colors[GFP[1],2] = 0
    Colors_nucleus = Colors.copy()
    return Colors, Colors_nucleus

def coloring_function_KI67_stain(df_cells):
    Cell_phase = np.array([df_cells['current_phase']])
    Gene_RFP = np.array([df_cells['genes_0']])
    Ki67p = np.where(Cell_phase == 2)
    Ki67n = np.where(Cell_phase != 2)
    RFP = np.where(Gene_RFP == 1)
    GFP = np.where(Gene_RFP == 0)
    # Colors
    Colors = np.ones((len(df_cells),4)) # 4 channels RGBO
    Colors_nucleus = np.ones((len(df_cells),4)) # 4 channels RGBO
    # Ki67+ cells - Nucleus white
    Colors_nucleus[Ki67p[1],0] = 1
    Colors_nucleus[Ki67p[1],1] = 1
    Colors_nucleus[Ki67p[1],2] = 1
    # Ki67- cells - Nucleus black
    Colors_nucleus[Ki67n[1],0] = 0
    Colors_nucleus[Ki67n[1],1] = 0
    Colors_nucleus[Ki67n[1],2] = 0
    # RFP+ cells - Cytoplasm red
    Colors[RFP[1],0] = 1
    Colors[RFP[1],1] = 0
    Colors[RFP[1],2] = 0
    # GFP+ cells - Cytoplasm green
    Colors[GFP[1],0]= 0
    Colors[GFP[1],1] = 1
    Colors[GFP[1],2] = 0
    return Colors, Colors_nucleus

def my_custom_header_function(mcds):
    # Current time
    curr_time = round(mcds.get_time(),2) # min
    time_days = curr_time//1440.0
    time_hours = (curr_time%1440.0)//60
    time_min = ((curr_time%1440.0)%60)
    df_Cells = mcds.get_cell_df()
    # Selecting RFP+ and GFP+ cells
    Gene_RFP = np.array([df_Cells['genes_0']])
    RFP = np.where(Gene_RFP == 1)
    GFP = np.where(Gene_RFP == 0)
    Count_RFP = len(RFP[1])
    Count_GFP = len(GFP[1])
    title_text = "Current time: %02d days, %02d hours, and %0.2f minutes \n\n RFP+: %d agents and GFP+: %d agents"%(time_days,time_hours,time_min,Count_RFP,Count_GFP)
    return title_text

if __name__ == '__main__':
    if (len(sys.argv) != 3 and len(sys.argv) != 2):
      print("Please provide\n 1 arg [folder]: to taking snapshots from the folder \n or provide 2 args [folder] [frame ID]: to interact with scene!")
      sys.exit(1)
    if (len(sys.argv) == 3):
      # AddBox = {'xmin': -250, 'xmax': 250, 'ymin': -250, 'ymax': 250, 'zmin': 500, 'zmax': 1000} # output_H1 (500 microns^3)
      # PlaneYZ1 = -97
      # PlaneYZ2 = 147

      AddBox = {'xmin': 500, 'xmax': 1000, 'ymin': 500, 'ymax': 1000, 'zmin': -960, 'zmax': -460} # output_H2 (500 microns^3)
      PlaneYZ1 = 580
      PlaneYZ2 = 870

      folder = Path(sys.argv[1])
      file = "/output%08d.xml"%int(sys.argv[2])
      CreateScene(folder,file,pc.pyMCDS(str(folder)+file, graph=False), coloring_function=my_custom_coloring_function, header_function=my_custom_header_function, PlotNucleus=True)
      # CreateScene(folder,file,pc.pyMCDS(str(folder)+file, graph=False), coloring_function=my_custom_coloring_function, header_function=my_custom_header_function, BoxCrop = AddBox,PlotNucleus=True)

      # SNAPSHOTS PostHypoxia stain
      # CreateSnapshots(folder, coloring_function=my_custom_coloring_function, header_function=my_custom_header_function,size_window=(5000,5000), PlotNucleus=True, AddBox = AddBox, file=file, add_name='_Box')
      # CreateSnapshots(folder, coloring_function=my_custom_coloring_function, header_function=my_custom_header_function,size_window=(5000,5000), PlotNucleus=True, BoxCrop = AddBox, file=file, add_name='_BoxCrop')
      # SNAPSHOTS Ki67 stain
      # CreateSnapshots(folder, coloring_function=coloring_function_KI67_stain, header_function=my_custom_header_function,size_window=(5000,5000), PlotNucleus=True, AddBox = AddBox, file=file, add_name='_Box_Ki67')
      # CreateSnapshots(folder, coloring_function=coloring_function_KI67_stain, header_function=my_custom_header_function,size_window=(5000,5000), PlotNucleus=True, BoxCrop = AddBox, file=file, add_name='_BoxCrop_Ki67', PlaneYZ_1=PlaneYZ1, PlaneYZ_2=PlaneYZ2)
    if (len(sys.argv) == 2):
      CreateSnapshots(Path(sys.argv[1]), coloring_function=my_custom_coloring_function, header_function=my_custom_header_function, size_window=(5000,5000), PlotNucleus=True)
