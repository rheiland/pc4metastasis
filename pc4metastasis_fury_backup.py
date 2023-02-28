#!/usr/bin/env python
# coding: utf-8

# In[1]:


style = """
    <style>
       .jupyter-widgets-output-area .output_scroll {
            height: unset !important;
            border-radius: unset !important;
            -webkit-box-shadow: unset !important;
            box-shadow: unset !important;
        }
        .jupyter-widgets-output-area  {
            height: auto !important;
        }
    </style>
    """


# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')
from IPython.core.display import HTML
from IPython.display import display
display(HTML("<style>.container { width:100% !important; }</style>"))
display(HTML(style))


# In[3]:


import sys, os
sys.path.insert(0, os.path.abspath('bin'))
#os.environ['CACHEDIR'] = os.path.join(os.path.abspath(''), 'tmpdir')
from DataSet import *

GUI_ModeSelectionData()


# In[4]:


#%run My_fury3D.py output_H1 80


# In[18]:


from fury_tab import FuryTab

fury_tab = FuryTab()
tab_height = 'auto'
tab_layout = widgets.Layout(width='auto',height=tab_height, overflow_y='scroll',)
titles = ['Fury']
tabs = widgets.Tab(children=[fury_tab.tab],
                   _titles={i: t for i, t in enumerate(titles)},
                   layout=tab_layout)
fury_feedback_str = widgets.Label(value='')

def send_data_to_fury_cb(b):
    fury_feedback_str.value = "working..."
    session_dir = os.getenv('SESSIONDIR')
    print('session_dir = ',session_dir)
    session_id = os.getenv('SESSION')
    print('session_id = ',session_id)
    user_id = os.getenv('USER')
    print('user_id = ',user_id)
    fury_data_path_str = "/data/tools/shared/" + user_id + "/fury/" + session_id
    # updated, based on email from Serge (1/19/21)
    fury_data_path_str2 = "/srv/nanohub/data/tools/shared/" + user_id + "/fury/" + session_id

    # dummy to test locally
    # fury_data_path_str = "/tmp/" + user_id + "/fury" 
    print("fury_data_path_str = ",fury_data_path_str)
    print("fury_data_path_str2 = ",fury_data_path_str2)

    os.makedirs(fury_data_path_str, exist_ok=True)
    # data_file = "output00000001_cells_physicell.mat"

    # we need to copy 3(?) files (for any one frame)
    mesh_file = "initial_mesh0.mat" 
#     frame_num = 13
    frame_num = 80
    xml_file = "output%08d.xml" % frame_num
    data_file = "output%08d_cells_physicell.mat" % frame_num
    # from the app's root directory
    # print("self.output_dir = ",self.output_dir)
    # from_file = "tmpdir/" + data_file

    # from_file = self.output_dir + "/" + mesh_file
    from_file = "data/" + mesh_file
    to_file = fury_data_path_str + "/" + mesh_file
    copyfile(from_file, to_file)

    # from_file = self.output_dir + "/" + xml_file
    from_file = "data/" + xml_file
    to_file = fury_data_path_str + "/" + xml_file
    copyfile(from_file, to_file)

    # from_file = self.output_dir + "/" + data_file
    from_file = "data/" + data_file
    print("from: ",from_file)
    to_file = fury_data_path_str + "/" + data_file
    print("to: ",to_file)
    copyfile(from_file, to_file)

#                time.sleep(3)
    file = Path(to_file)
    while not file.exists():
        time.sleep(2)

    # copyfile("tmpdir/" + data_file, fury_data_path_str + "/" + "output00000001_cells_physicell.mat")

    # Send signal to Fury that new data is ready: (folder, filename)
    fury_tab.send_data(fury_data_path_str2, xml_file)

    fury_feedback_str.value = ""

#---------------------------
def send_local_data_to_fury_cb(b):
    fury_feedback_str.value = "working..."
    session_dir = os.getenv('SESSIONDIR')
    print('session_dir = ',session_dir)
    session_id = os.getenv('SESSION')
    print('session_id = ',session_id)
    user_id = os.getenv('USER')
    print('user_id = ',user_id)
    fury_data_path_str = "/data/tools/shared/" + user_id + "/fury/" + session_id
    # updated, based on email from Serge (1/19/21)
    fury_data_path_str2 = "/srv/nanohub/data/tools/shared/" + user_id + "/fury/" + session_id

    # dummy to test locally
    # fury_data_path_str = "/tmp/" + user_id + "/fury" 
    print("fury_data_path_str = ",fury_data_path_str)
    print("fury_data_path_str2 = ",fury_data_path_str2)

    os.makedirs(fury_data_path_str, exist_ok=True)
    # data_file = "output00000001_cells_physicell.mat"

    # we need to copy 3(?) files (for any one frame)
    mesh_file = "initial_mesh0.mat" 
#     frame_num = 13
    frame_num = 80
    xml_file = "output%08d.xml" % frame_num
    data_file = "output%08d_cells_physicell.mat" % frame_num
    # from the app's root directory
    # print("self.output_dir = ",self.output_dir)
    # from_file = "tmpdir/" + data_file

    # from_file = self.output_dir + "/" + mesh_file
    from_file = "data/" + mesh_file
    to_file = fury_data_path_str + "/" + mesh_file
    copyfile(from_file, to_file)

    # from_file = self.output_dir + "/" + xml_file
    from_file = "data/" + xml_file
    to_file = fury_data_path_str + "/" + xml_file
    copyfile(from_file, to_file)

    # from_file = self.output_dir + "/" + data_file
    from_file = "data/" + data_file
    print("from: ",from_file)
    to_file = fury_data_path_str + "/" + data_file
    print("to: ",to_file)
    copyfile(from_file, to_file)

#                time.sleep(3)
    file = Path(to_file)
    while not file.exists():
        time.sleep(2)

    # copyfile("tmpdir/" + data_file, fury_data_path_str + "/" + "output00000001_cells_physicell.mat")

    # Send signal to Fury that new data is ready: (folder, filename)
    fury_tab.send_data(fury_data_path_str2, xml_file)

    fury_feedback_str.value = ""

#--------------------
fury_button= widgets.Button(
    description="Send data to Fury", #style={'description_width': 'initial'},
    button_style='success',  # 'success', 'info', 'warning', 'danger' or ''
    tooltip='Click to send data to the Fury GPU server',
    disabled=False,
    layout=widgets.Layout(width='280px')
)
fury_button.on_click(send_data_to_fury_cb)
# fury_button.on_click(send_local_data_to_fury_cb)


gui = widgets.VBox(children=[fury_button, fury_feedback_str, tabs])
gui


# In[ ]:




