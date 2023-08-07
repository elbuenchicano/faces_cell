import  numpy as np
import  pandas as pd

import  matplotlib.pyplot as plt
import  matplotlib.colors as mcolors
from    matplotlib.colors import ListedColormap, LinearSegmentedColormap
import  plotly.express as px
from    plotly.subplots import make_subplots
import  plotly.graph_objects as go
import  random


'''
t.f creates a 3d chart
'''
def chart(X, y):
    #--------------------------------------------------------------------------#
    # This section is not mandatory as its purpose is to sort the data by label 
    # so, we can maintain consistent colors for digits across multiple graphs
    
    # Concatenate X and y arrays
    arr_concat=np.concatenate((X, y.reshape(y.shape[0],1)), axis=1)
    # Create a Pandas dataframe using the above array
    df=pd.DataFrame(arr_concat, columns=['x', 'y', 'z', 'label'])
    # Convert label data type from float to integer
    df['x'] = df['x'].astype(float)
    df['y'] = df['y'].astype(float)
    df['z'] = df['z'].astype(float)
    # Finally, sort the dataframe by label
    #df.sort_values(by='label', axis=0, ascending=True, inplace=True)
    #--------------------------------------------------------------------------#
    
    # Create a 3D graph
    fig = px.scatter_3d(df, x='x', y='y', z='z', color=df['label'].astype(str), height=900, width=950)

    # Update chart looks
    fig.update_layout(title_text='UMAP',
                      showlegend=True,
                      legend=dict(orientation="h", yanchor="top", y=0, xanchor="center", x=0.5),
                      scene_camera=dict(up=dict(x=0, y=0, z=1), 
                                            center=dict(x=0, y=0, z=-0.1),
                                            eye=dict(x=1.5, y=-1.4, z=0.5)),
                                            margin=dict(l=0, r=0, b=0, t=0),
                      scene = dict(xaxis=dict(backgroundcolor='white',
                                              color='black',
                                              gridcolor='#f0f0f0',
                                              title_font=dict(size=10),
                                              tickfont=dict(size=10),
                                             ),
                                   yaxis=dict(backgroundcolor='white',
                                              color='black',
                                              gridcolor='#f0f0f0',
                                              title_font=dict(size=10),
                                              tickfont=dict(size=10),
                                              ),
                                   zaxis=dict(backgroundcolor='lightgrey',
                                              color='black', 
                                              gridcolor='#f0f0f0',
                                              title_font=dict(size=10),
                                              tickfont=dict(size=10),
                                             )))
    # Update marker size
    fig.update_traces(marker=dict(size=3, line=dict(color='black', width=0.1)))
    
    fig.show()


def scatter2D(x,y):
    arr_concat=np.concatenate((x, y.reshape(y.shape[0],1)), axis=1)
    # Create a Pandas dataframe using the above array
    df=pd.DataFrame(arr_concat, columns=['x', 'y', 'label'])
    
    df['x'] = df['x'].astype(float)
    df['y'] = df['y'].astype(float)
    

    fig = px.scatter(df, x="x", y="y", color=df["label"].astype(str))
    fig.show()


################################################################################
################################################################################
################################################################################
'''
T.f. defines a colormap manually using variable n
'''
def colorMapByNumber(n):
    hexadecimal_alphabets = '0123456789ABCDEF'
    cm = ["#" + ''.join([random.choice(hexadecimal_alphabets) for j in
                range(6)]) for i in range(n)]
    return ListedColormap(cm)

