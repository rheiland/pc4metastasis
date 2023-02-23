"""
Provide simple plotting functionality for PhysiCell output results in 3D.

Authors:
Heber L Rocha (hlimadar@iu.edu)
Furkan Kurtoglu(fkurtog@iu.edu)
"""

import pcDataLoader as pc
from pathlib import Path
import numpy as np
from fury import window, actor, ui
from fury.data import read_viz_icons, fetch_viz_icons
import pyvista as pv
import fury.utils as ut_vtk
import sys,os

def coloring_function_default(df_cells):
    Cell_types = df_cells['cell_type'].unique()
    # Default colors (default color will be grey )
    Colors_default = np.array([[255,87,51], [255,0,0], [255,211,0], [0,255,0], [0,0,255], [254,51,139], [255,131,0], [50,205,50], [0,255,255], [255,0,127], [255,218,185], [143,188,143], [135,206,250]])/255.0 # grey, red, yellow, green, blue, magenta, orange, lime, cyan, hotpink, peachpuff, darkseagreen, lightskyblue
    apoptotic_color = np.array([255,255,255])/255.0 # white
    necrotic_color = np.array([139,69,19])/255.0 # brown
    cells_type = np.array([df_cells['cell_type']])
    cell_cycle = np.array([df_cells['cycle_model']])
    # Colors
    Colors = np.ones((len(df_cells),4)) # 4 channels RGBO
    for index,type in enumerate(Cell_types):
        idxs = np.where( (cells_type == type) & (cell_cycle < 100) )
        Colors[idxs,0] = Colors_default[index,0]
        Colors[idxs,1] = Colors_default[index,1]
        Colors[idxs,2] = Colors_default[index,2]
        idxs_apoptotic = np.where( cell_cycle == 100 )
        Colors[idxs_apoptotic,0] = apoptotic_color[0]
        Colors[idxs_apoptotic,1] = apoptotic_color[1]
        Colors[idxs_apoptotic,2] = apoptotic_color[2]
        idxs_necrotic = np.where( cell_cycle > 100 )
        Colors[idxs_necrotic,0] = necrotic_color[0]
        Colors[idxs_necrotic,1] = necrotic_color[1]
        Colors[idxs_necrotic,2] = necrotic_color[2]
    Colors_nucleus = Colors.copy()
    return Colors, Colors_nucleus

def header_function_default(mcds):
    # Current time
    curr_time = round(mcds.get_time(),2) # min
    time_days = curr_time//1440.0
    time_hours = (curr_time%1440.0)//60
    time_min = ((curr_time%1440.0)%60)
    # Number of cells
    Num_cells = len(mcds.get_cell_df())
    title_text = "Current time: %02d days, %02d hours, and %0.2f minutes, %d agents"%(time_days,time_hours,time_min,Num_cells)
    return title_text

def DrawBox(Bounds, Color, scaling):
    lines = [np.array([[Bounds['xmin'],Bounds['ymin'],Bounds['zmin']],[Bounds['xmax'],Bounds['ymin'],Bounds['zmin']],[Bounds['xmax'],Bounds['ymax'],Bounds['zmin']],[Bounds['xmin'],Bounds['ymax'],Bounds['zmin']],[Bounds['xmin'],Bounds['ymin'],Bounds['zmin']],[Bounds['xmin'],Bounds['ymin'],Bounds['zmax']],[Bounds['xmin'],Bounds['ymax'],Bounds['zmax']],[Bounds['xmin'],Bounds['ymax'],Bounds['zmin']],[Bounds['xmin'],Bounds['ymax'],Bounds['zmax']],[Bounds['xmax'],Bounds['ymax'],Bounds['zmax']],[Bounds['xmax'],Bounds['ymax'],Bounds['zmin']],[Bounds['xmax'],Bounds['ymax'],Bounds['zmin']],[Bounds['xmax'],Bounds['ymin'],Bounds['zmin']],[Bounds['xmax'],Bounds['ymin'],Bounds['zmax']],[Bounds['xmax'],Bounds['ymax'],Bounds['zmax']],[Bounds['xmax'],Bounds['ymin'],Bounds['zmax']],[Bounds['xmin'],Bounds['ymin'],Bounds['zmax']]])]
    return actor.line(lines, Color, linewidth=0.003*scaling)

def CreateScene(folder, InputFile, mcds, coloring_function = coloring_function_default, header_function = header_function_default, size_window=(1000,1000), FileName=None, BoxCrop = None, AddBox = None, PlotNucleus=False, PlaneXY_1=None, PlaneXY_2=None, PlaneXZ_1=None, PlaneXZ_2=None, PlaneYZ_1=None, PlaneYZ_2=None, saveVTK =False ):
    # Define domain size
    domain_range = mcds.get_xyz_range()
    dx, dy, dz = mcds.get_mesh_spacing()
    Bounds = {'xmin': domain_range[0][0], 'xmax': domain_range[0][1], 'ymin': domain_range[1][0], 'ymax': domain_range[1][1], 'zmin': domain_range[2][0], 'zmax': domain_range[2][1]}
    # Cell positions
    df_Cells = mcds.get_cell_df()
    if ( BoxCrop ):
        Bounds = BoxCrop
        df_Cells = df_Cells.loc[ (df_Cells['position_x'] > BoxCrop['xmin']) & (df_Cells['position_x'] < BoxCrop['xmax']) & (df_Cells['position_y'] > BoxCrop['ymin']) & (df_Cells['position_y'] < BoxCrop['ymax']) & (df_Cells['position_z'] > BoxCrop['zmin']) & (df_Cells['position_z'] < BoxCrop['zmax']) ]
    # Define boundaries of domain
    x_min_domain = Bounds['xmin']; y_min_domain = Bounds['ymin']; z_min_domain = Bounds['zmin']
    x_max_domain = Bounds['xmax']; y_max_domain = Bounds['ymax']; z_max_domain = Bounds['zmax']
    # Cell positions, radius, and colors
    C_xyz = np.zeros((len(df_Cells),3))
    C_xyz[:,0] =  df_Cells['position_x']
    C_xyz[:,1] =  df_Cells['position_y']
    C_xyz[:,2] =  df_Cells['position_z']
    # Cytoplasm Radius Calculation
    C_radii = np.cbrt(df_Cells['total_volume'] * 0.75 / np.pi).to_numpy() # r = np.cbrt(V * 0.75 / pi)
    # Cytoplasm Radius Calculation
    C_radii_nucleus = np.cbrt(df_Cells['nuclear_volume'] * 0.75 / np.pi).to_numpy()
    # Coloring
    C_colors, C_colors_nucleus = coloring_function(df_Cells)

    ###############################################################################################
    # Creaating Scene
    scaling = size_window[0]
    showm = window.ShowManager(size=size_window, reset_camera=True, order_transparent=True, title="PhysiCell Fury: "+InputFile)
    domain_size = ( x_max_domain-x_min_domain, y_max_domain-y_min_domain, z_max_domain-z_min_domain )
    domain_center = ( x_min_domain + 0.5*domain_size[0],  y_min_domain + 0.5*domain_size[1],  z_min_domain + 0.5*domain_size[2] )
    camera_position = ( domain_center[0] -1.375*domain_size[0], domain_center[1], domain_center[2] + 3.5*domain_size[2] )
    camera_focalpoint = domain_center
    camera_viewup = (0,1,0)
    # showm.scene.SetBackground((1,1,1)) Background color
    ###############################################################################################
    # TITLE
    title_text = header_function(mcds)
    title = ui.TextBlock2D(text=title_text, font_size=scaling//50, font_family='Arial', justification='center', vertical_justification='bottom', bold=False, italic=False, shadow=False, color=(1, 1, 1), bg_color=None, position=(round(0.5*scaling), round(0.9*scaling)))
    showm.scene.add(title)
    GrayColor = np.array([0.5,0.5,0.5])
    WhiteColor = np.array([1.0,1.0,1.0])
    # Drawing Domain Boundaries
    showm.scene.add(DrawBox(Bounds, GrayColor, scaling))
    # Drawing Additional Box
    if ( AddBox ): showm.scene.add(DrawBox(AddBox, WhiteColor, scaling))
    ###############################################################################################
    # Creating Sphere Actor for all cells
    if ( PlotNucleus ):
        C_colors[:,-1] = 0.3 # transparency on cytoplasm
        sphere_actor_nucleus = actor.sphere(centers=C_xyz,colors=C_colors_nucleus,radii=C_radii_nucleus) # Nucleus
        showm.scene.add(sphere_actor_nucleus)
    sphere_actor = actor.sphere(centers=C_xyz,colors=C_colors,radii=C_radii) # Cytoplasm
    showm.scene.add(sphere_actor)
    sphere_actor_cutted = None
    sphere_actor_cutted_nucleus = None
    flag_cut = False
    ###############################################################################################
    # Planes sections
    if PlaneXY_1: box_actorXY_min = actor.box(np.array([[domain_center[0],domain_center[1],PlaneXY_1]]), np.array([[1,1,0]]), colors=(0.5, 0.5, 0.5,0.4),scales=(domain_size[0],domain_size[1],0.5*dz))
    else: box_actorXY_min = actor.box(np.array([[domain_center[0],domain_center[1],z_min_domain]]), np.array([[1,1,0]]), colors=(0.5, 0.5,0.5,0.4),scales=(domain_size[0],domain_size[1],0.5*dz))
    if PlaneXY_2: box_actorXY_max = actor.box(np.array([[domain_center[0],domain_center[1],PlaneXY_2]]), np.array([[1,1,0]]), colors=(0.5, 0.5, 0.5,0.4),scales=(domain_size[0],domain_size[1],0.5*dz))
    else: box_actorXY_max = actor.box(np.array([[domain_center[0],domain_center[1],z_max_domain]]), np.array([[1,1,0]]), colors=(0.5, 0.5, 0.5,0.4),scales=(domain_size[0],domain_size[1],0.5*dz))
    if PlaneXZ_1: box_actorXZ_min = actor.box(np.array([[domain_center[0],PlaneXZ_1,domain_center[2]]]), np.array([[1,0,1]]), colors=(0.5, 0.5, 0.5,0.4),scales=(domain_size[0],0.5*dy,domain_size[2]))
    else: box_actorXZ_min = actor.box(np.array([[domain_center[0],y_min_domain,domain_center[2]]]), np.array([[1,0,1]]), colors=(0.5, 0.5, 0.5,0.4),scales=(domain_size[0],0.5*dy,domain_size[2]))
    if PlaneXZ_2: box_actorXZ_max = actor.box(np.array([[domain_center[0],PlaneXZ_2,domain_center[2]]]), np.array([[1,0,1]]), colors=(0.5, 0.5, 0.5,0.4),scales=(domain_size[0],0.5*dy,domain_size[2]))
    else: box_actorXZ_max = actor.box(np.array([[domain_center[0],y_max_domain,domain_center[2]]]), np.array([[1,0,1]]), colors=(0.5, 0.5, 0.5,0.4),scales=(domain_size[0],0.5*dy,domain_size[2]))
    if PlaneYZ_1: box_actorYZ_min = actor.box(np.array([[PlaneYZ_1,domain_center[1],domain_center[2]]]), np.array([[0,1,1]]), colors=(0.9, 0.9, 0.9,0.4),scales=(0.5*dx,domain_size[1],domain_size[2]))
    else: box_actorYZ_min = actor.box(np.array([[x_min_domain,domain_center[1],domain_center[2]]]), np.array([[0,1,1]]), colors=(0.5, 0.5, 0.5,0.4),scales=(0.5*dx,domain_size[1],domain_size[2]))
    if PlaneYZ_2: box_actorYZ_max = actor.box(np.array([[PlaneYZ_2,domain_center[1],domain_center[2]]]), np.array([[0,1,1]]), colors=(0.9, 0.9, 0.9,0.4),scales=(0.5*dx,domain_size[1],domain_size[2]))
    else: box_actorYZ_max = actor.box(np.array([[x_max_domain,domain_center[1],domain_center[2]]]), np.array([[0,1,1]]), colors=(0.5, 0.5, 0.5,0.4),scales=(0.5*dx,domain_size[1],domain_size[2]))
    showm.scene.add(box_actorXY_min)
    showm.scene.add(box_actorXY_max)
    showm.scene.add(box_actorXZ_min)
    showm.scene.add(box_actorXZ_max)
    showm.scene.add(box_actorYZ_min)
    showm.scene.add(box_actorYZ_max)
    # Section in plane XY
    line_slider_xy = ui.LineDoubleSlider2D(center=(round(0.2*scaling), round(0.05*scaling)),initial_values=(z_min_domain,z_max_domain), min_value=z_min_domain, max_value=z_max_domain, orientation="horizontal", font_size=scaling//50)
    showm.scene.add(line_slider_xy)
    def translate_planeXY(slider):
        plane_min = np.array([0,0, slider.left_disk_value-z_min_domain])
        plane_max = np.array([0,0, slider.right_disk_value-z_max_domain])
        box_actorXY_min.SetPosition(plane_min)
        box_actorXY_max.SetPosition(plane_max)
    line_slider_xy.on_change = translate_planeXY
    line_slider_xy_label = ui.TextBlock2D(position=(round(0.35*scaling),round(0.04*scaling)),text="plane XY (microns)", font_size=scaling//50)
    showm.scene.add(line_slider_xy_label)
    # Section in plane XZ
    line_slider_xz = ui.LineDoubleSlider2D(center=(round(0.2*scaling), round(0.1*scaling)), initial_values=(y_min_domain,y_max_domain), min_value=y_min_domain, max_value=y_max_domain, orientation="horizontal", font_size=scaling//50)
    showm.scene.add(line_slider_xz)
    def translate_planeXZ(slider):
        plane_min = np.array([0,slider.left_disk_value-y_min_domain,0])
        plane_max = np.array([0,slider.right_disk_value-y_max_domain,0])
        box_actorXZ_min.SetPosition(plane_min)
        box_actorXZ_max.SetPosition(plane_max)
    line_slider_xz.on_change = translate_planeXZ
    line_slider_xz_label = ui.TextBlock2D(position=(round(0.35*scaling),round(0.09*scaling)),text="plane XZ (microns)", font_size=scaling//50)
    showm.scene.add(line_slider_xz_label)
    # Section in plane YZ
    line_slider_yz = ui.LineDoubleSlider2D(center=(round(0.2*scaling), round(0.15*scaling)), initial_values=(x_min_domain,x_max_domain), min_value=x_min_domain, max_value=x_max_domain, orientation="horizontal", font_size=scaling//50)
    line_slider_yz.default_color = (1,0,0) # Red
    showm.scene.add(line_slider_yz)
    def translate_planeYZ(slider):
        plane_min = np.array([slider.left_disk_value-x_min_domain,0,0])
        plane_max = np.array([slider.right_disk_value-x_max_domain,0,0])
        box_actorYZ_min.SetPosition(plane_min)
        box_actorYZ_max.SetPosition(plane_max)
    line_slider_yz.on_change = translate_planeYZ
    line_slider_yz_label = ui.TextBlock2D(position=(round(0.35*scaling),round(0.14*scaling)),text="plane YZ (microns)", font_size=scaling//50)
    showm.scene.add(line_slider_yz_label)
    ###############################################################################################
    # Button to slice
    def AddCells():
        global sphere_actor_cutted, sphere_actor_cutted_nucleus
        idx_cells = np.argwhere( (C_xyz[:,2] > line_slider_xy.left_disk_value) & ( C_xyz[:,2] < line_slider_xy.right_disk_value) & (C_xyz[:,1] > line_slider_xz.left_disk_value) & ( C_xyz[:,1] < line_slider_xz.right_disk_value) & (C_xyz[:,0] > line_slider_yz.left_disk_value) & ( C_xyz[:,0] < line_slider_yz.right_disk_value) ).flatten()
        sphere_actor_cutted = actor.sphere(centers=C_xyz[idx_cells,:],colors=C_colors[idx_cells,:],radii=C_radii[idx_cells]) # Cytoplasm
        sphere_actor_cutted_nucleus = actor.sphere(centers=C_xyz[idx_cells,:],colors=C_colors_nucleus[idx_cells,:],radii=C_radii_nucleus[idx_cells]) # Nucleus
        if ( PlotNucleus ):
            showm.scene.add(sphere_actor_cutted_nucleus)
        showm.scene.add(sphere_actor_cutted)
        return idx_cells.shape[0]
    def SliceCells(i_ren, _obj, _button):
        global sphere_actor_cutted,sphere_actor_cutted_nucleus,flag_cut
        if (button_slice_label.message == 'Cut'):
            # Clear up the cells
            showm.scene.rm(sphere_actor)
            if ( PlotNucleus ): showm.scene.rm(sphere_actor_nucleus)
            # Selecting the cells in the region
            NumCells = AddCells()
            print("------------------------------------------------------------------\n Cut")
            print(f"Plane XY - Min:{line_slider_xy.left_disk_value} Max:{line_slider_xy.right_disk_value}")
            print(f"Plane XZ - Min:{line_slider_xz.left_disk_value} Max:{line_slider_xz.right_disk_value}")
            print(f"Plane YZ - Min:{line_slider_yz.left_disk_value} Max:{line_slider_yz.right_disk_value}")
            print(f"Number of cells: {NumCells}")
            print("------------------------------------------------------------------")
            button_slice_label.message = ' Cut ' #Check out a more elegant way!
            flag_cut = True
        else:
            # Clear up the cells
            showm.scene.rm(sphere_actor_cutted)
            if ( PlotNucleus ): showm.scene.rm(sphere_actor_cutted_nucleus)
            # Selecting the cells in the region
            NumCells = AddCells()
            print("------------------------------------------------------------------\n Cut")
            print(f"Plane XY - Min:{line_slider_xy.left_disk_value} Max:{line_slider_xy.right_disk_value}")
            print(f"Plane XZ - Min:{line_slider_xz.left_disk_value} Max:{line_slider_xz.right_disk_value}")
            print(f"Plane YZ - Min:{line_slider_yz.left_disk_value} Max:{line_slider_yz.right_disk_value}")
            print(f"Number of cells: {NumCells}")
            print("------------------------------------------------------------------")
        hide_all_widgets()
        i_ren.force_render()
    button_slice_label = ui.TextBlock2D(text="Cut", font_size=scaling//50, font_family='Arial', justification='center', vertical_justification='middle', bold=True, italic=False, shadow=False, color=(1, 1, 1), bg_color=None, position=(round(0.6*scaling), round(0.1*scaling)))
    # First we need to fetch some icons that are included in FURY.
    fetch_viz_icons()
    button_slice = ui.Button2D(icon_fnames=[('square',read_viz_icons(fname="stop2.png"))],size=(round(0.1*scaling),round(0.05*scaling)) ,position=(round(0.55*scaling),round(0.075*scaling)))
    button_slice.on_left_mouse_button_clicked = SliceCells
    showm.scene.add(button_slice)
    showm.scene.add(button_slice_label)
    ###############################################################################################
    # Add referencial vectors axis
    center = np.array([[x_max_domain,y_min_domain,z_max_domain],[x_max_domain,y_min_domain,z_max_domain],[x_max_domain,y_min_domain,z_max_domain]])
    x_direction = np.array([-1,0,0])
    y_direction = np.array([0,1,0])
    z_direction = np.array([0,0,-1])
    direction_arrow = np.array([x_direction,y_direction,z_direction])
    arrow_actor = actor.arrow(center,direction_arrow,np.array([[1,0,0],[0,1,0],[0,0,1]]),heights=0.25*min(domain_size),tip_radius=0.1)
    showm.scene.add(arrow_actor)
    # Substrates
    substrates = mcds.get_substrate_names()
    substrate_combobox = ui.ComboBox2D(items=substrates, placeholder="Choose substrate", position=(round(0.15*scaling), round(0.05*scaling)), size=(round(0.5*scaling), round(0.1*scaling)), font_size=scaling//50)
    # Preparing pyvista mesh
    substrate0 = np.transpose(mcds.get_concentration(substrates[0])) # CHECK ORDER ON PYMCDS
    # Create the spatial reference
    grid = pv.UniformGrid()
    # Set the grid dimensions: shape + 1 because we want to inject our values on
    # the CELL data (mesh)
    grid.dimensions = np.array(substrate0.shape) + 1
    # Edit the spatial reference
    grid.origin = (x_min_domain, y_min_domain, z_min_domain)  # The bottom left corner of the data set
    grid.spacing = (dx, dy, dz)  # These are the cell sizes along each axis
    # Add the data values to the cell data
    grid.cell_data[str(substrates[0])] = substrate0.flatten(order="F")  # Flatten the array!
    pv.global_theme.cmap = 'coolwarm_r' # cmap palette color
    # Add others Substrates
    for i in range(1,len(substrates)):
        grid.cell_data.set_array( np.transpose(mcds.get_concentration(substrates[i])).flatten(order="F"),substrates[i]) # CHECK ORDER ON PYMCDS

    def change_substrate(combobox):
        selected_substrate = combobox.selected_text
        grid.cell_data.active_scalars_name = selected_substrate # select substrate on grid
        # p = grid.plot(show_edges=True)
        p = pv.Plotter()
        p.add_text(selected_substrate)
        boxwidgets = p.add_mesh_clip_box(grid)
        # print(boxwidgets.box_clipped_meshes)
        p.add_camera_orientation_widget()
        #print(p.box_clipped_meshes)
        p.show()
        # print(grid)
    substrate_combobox.on_change = change_substrate
    showm.scene.add(substrate_combobox)
    ###############################################################################################
    # Menu list
    MenuValues = ['Substrates','Cutting plane XY','Cutting plane XZ','Cutting plane YZ','Reset Camera','Reset','Snapshot']
    Actors = [[substrate_combobox],[button_slice,button_slice_label],[line_slider_xy,line_slider_xy_label],[line_slider_xz,line_slider_xz_label],[line_slider_yz,line_slider_yz_label]]
    listbox = ui.ListBox2D(values=MenuValues, position=(round(0.7*scaling), 0), size=(round(0.3*scaling), round(0.2*scaling)), multiselection=False, font_size=scaling//50)
    def hide_all_widgets():
        for actor  in Actors:
            for element in actor:
                element.set_visibility(False)
        if PlaneXY_1: box_actorXY_min.SetVisibility(True)
        else: box_actorXY_min.SetVisibility(False)
        if PlaneXY_2: box_actorXY_max.SetVisibility(True)
        else: box_actorXY_max.SetVisibility(False)
        if PlaneXZ_1: box_actorXZ_min.SetVisibility(True)
        else: box_actorXZ_min.SetVisibility(False)
        if PlaneXZ_2: box_actorXZ_max.SetVisibility(True)
        else: box_actorXZ_max.SetVisibility(False)
        if PlaneYZ_1: box_actorYZ_min.SetVisibility(True)
        else: box_actorYZ_min.SetVisibility(False)
        if PlaneYZ_2: box_actorYZ_max.SetVisibility(True)
        else: box_actorYZ_max.SetVisibility(False)
    hide_all_widgets()
    def MenuOption():
        hide_all_widgets()
        if MenuValues[MenuValues.index(listbox.selected[0])] == 'Substrates':
            substrate_combobox.set_visibility(True)
        if MenuValues[MenuValues.index(listbox.selected[0])] == 'Cutting plane XY':
            box_actorXY_min.SetVisibility(True)
            box_actorXY_max.SetVisibility(True)
            line_slider_xy.set_visibility(True)
            line_slider_xy_label.set_visibility(True)
            button_slice.set_visibility(True)
            button_slice_label.set_visibility(True)
        if MenuValues[MenuValues.index(listbox.selected[0])] == 'Cutting plane XZ':
            box_actorXZ_min.SetVisibility(True)
            box_actorXZ_max.SetVisibility(True)
            line_slider_xz.set_visibility(True)
            line_slider_xz_label.set_visibility(True)
            button_slice.set_visibility(True)
            button_slice_label.set_visibility(True)
        if MenuValues[MenuValues.index(listbox.selected[0])] == 'Cutting plane YZ':
            box_actorYZ_min.SetVisibility(True)
            box_actorYZ_max.SetVisibility(True)
            line_slider_yz.set_visibility(True)
            line_slider_yz_label.set_visibility(True)
            button_slice.set_visibility(True)
            button_slice_label.set_visibility(True)
        if MenuValues[MenuValues.index(listbox.selected[0])] == 'Reset Camera':
            showm.scene.set_camera( position=camera_position, focal_point=camera_focalpoint, view_up=camera_viewup )
            # print(showm.scene.get_camera())
        if MenuValues[MenuValues.index(listbox.selected[0])] == 'Reset':
            line_slider_xy.left_disk_value = z_min_domain
            line_slider_xy.right_disk_value = z_max_domain
            line_slider_xz.left_disk_value = y_min_domain
            line_slider_xz.right_disk_value = y_max_domain
            line_slider_yz.left_disk_value = x_min_domain
            line_slider_yz.right_disk_value = x_max_domain
            global sphere_actor_cutted,sphere_actor_cutted_nucleus,flag_cut
            if (flag_cut):
                showm.scene.rm(sphere_actor_cutted)
                if ( PlotNucleus ): showm.scene.rm(sphere_actor_cutted_nucleus)
                AddCells()
        if MenuValues[MenuValues.index(listbox.selected[0])] == 'Snapshot':
            hide_all_widgets()
            listbox.set_visibility(False)
            FileName = "Snapshot_"+os.path.splitext(InputFile)[0]+".jpg"
            if ( type(folder) == str ):
                pathSave = folder+FileName
            else:
                pathSave = folder / FileName
            print("Genererated: ",pathSave)
            window.snapshot(showm.scene,order_transparent =True,size=(2*size_window[0],2*size_window[1]),fname=pathSave)
            listbox.set_visibility(True)
    listbox.on_change = MenuOption
    showm.scene.add(listbox)
    ###############################################################################################
    # Show Manager
    showm.scene.reset_camera()
    showm.scene.set_camera( position=camera_position, focal_point=camera_focalpoint, view_up=camera_viewup )
    # print(showm.scene.get_camera())
    ###############################################################################################
    # Save image
    if ( FileName ):
        hide_all_widgets()
        listbox.set_visibility(False)
        fileName = FileName+".jpg"
        fileNameVTK = FileName+".vtk"
        if ( type(folder) == str ):
            pathSave = folder+fileName
            pathSaveVTK = folder+fileNameVTK
        else:
            pathSave = str(folder)+fileName
            pathSaveVTK = str(folder)+fileNameVTK
        # window.snapshot(showm.scene,size=size_window,fname=pathSave)
        window.record(showm.scene, out_path=pathSave, size=size_window, reset_camera=False)
        if (saveVTK) :  grid.save(pathSaveVTK,binary=False) # save vtk file
        showm.scene.rm_all() # clean up the scene
    else:
        showm.start()

def CreateSnapshots(folder, coloring_function = coloring_function_default, header_function = header_function_default, size_window=(1000,1000), file=None, add_name=None, BoxCrop = None, AddBox = None, PlotNucleus=False, PlaneXY_1=None, PlaneXY_2=None, PlaneXZ_1=None, PlaneXZ_2=None, PlaneYZ_1=None, PlaneYZ_2=None):
    if (file):
        if (add_name): FileName = os.path.splitext(file)[0]+add_name
        else: FileName = os.path.splitext(file)[0]
        CreateScene(folder,file, pc.pyMCDS(str(folder)+file, graph=False),coloring_function=coloring_function, header_function=header_function,size_window=size_window,FileName=FileName, BoxCrop = BoxCrop, AddBox = AddBox, PlotNucleus=PlotNucleus,PlaneXY_1=PlaneXY_1,PlaneXY_2=PlaneXY_2,PlaneXZ_1=PlaneXZ_1,PlaneXZ_2=PlaneXZ_2,PlaneYZ_1=PlaneYZ_1,PlaneYZ_2=PlaneYZ_2)
    else:
        mcdsts = pc.pyMCDSts(folder, graph=False)  # generate a mcds time series instance
        ls_xml = mcdsts.get_xmlfile_list()
        l_mcds = mcdsts.read_mcds() # load all snapshots
        # Make snapshots
        for i, mcds in enumerate(l_mcds):
            if (add_name): FileName = "/output%08d"%i+add_name
            else: FileName = "/output%08d"%i
            CreateScene(folder,os.path.basename(ls_xml[i]),mcds,coloring_function=coloring_function, header_function=header_function,size_window=size_window,FileName=FileName, BoxCrop = BoxCrop, AddBox = AddBox, PlotNucleus=PlotNucleus,PlaneXY_1=PlaneXY_1,PlaneXY_2=PlaneXY_2,PlaneXZ_1=PlaneXZ_1,PlaneXZ_2=PlaneXZ_2,PlaneYZ_1=PlaneYZ_1,PlaneYZ_2=PlaneYZ_2)

if __name__ == '__main__':
    if (len(sys.argv) != 3 and len(sys.argv) != 2):
      print("Please provide\n 1 arg [folder]: to taking snapshots from the folder \n or provide 2 args [folder] [frame ID]: to interact with scene!")
      sys.exit(1)
    if (len(sys.argv) == 3):
        folder = Path(sys.argv[1])
        file = "/output%08d.xml"%int(sys.argv[2])
        CreateScene(folder,file,pc.pyMCDS(str(folder)+file, graph=False))
    if (len(sys.argv) == 2):
      CreateSnapshots(Path(sys.argv[1]))
