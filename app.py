import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import os



FIG_WIDTH = 800
FIG_HEIGHT = 800
FIG_OPACITY = 0.1
FONT_SIZE = 23
MARKER_SIZE = 20
DEFAULT_SELECTION = "Type name of the concept"
DEFAULT_NBR_NODETYPE_SELECTION = "All"
DEFAULT_FEATURE_NODETYPE_SELECTION = "Gene"
DEFAULT_NBR_COUNT = 20
MIN_NBR_COUNT = 5
MAX_NBR_COUNT = 50
MIN_FEATURE_COUNT = 5
MAX_FEATURE_COUNT = 50
DEFAULT_FEATURE_COUNT = 10

ROW_METADATA_PATH = os.path.join("data", "combined_beige_row_map_med_relevant_2_with_tsne_values.tsv")
COLUMN_METADATA_PATH = os.path.join("data", "combined_beige_column_map.tsv")
COSINE_DISTANCE_PATH = os.path.join("data", "cosine_distance_of_seps_combined_beige.npy")
MANHATTEN_DISTANCE_PATH = os.path.join("data", "manhatten_distance_of_seps_combined_beige.npy")
TOP_FEATURE_INDEX_PATH = os.path.join("data", "top_features_indices.npy")

def get_metadata():
    return pd.read_csv(ROW_METADATA_PATH, sep="\t")

def get_feature_metadata():
    return pd.read_csv(COLUMN_METADATA_PATH, sep="\t")

def get_dim():
    dim_list = ["2D", "3D"]
    return st.sidebar.radio("tSNE dimension", dim_list, index=0)

def get_search_term(metadata):
    return st.sidebar.selectbox("Search", list(metadata.sort_values(by="node_name_type").node_name_type.values) + [DEFAULT_SELECTION], index=len(metadata))

def get_nbr_nodetype(metadata):
    return st.sidebar.selectbox("Select the neighbor node type", list(metadata.sort_values(by="node_type").node_type.unique()) + [DEFAULT_NBR_NODETYPE_SELECTION], index=len(metadata.node_type.unique()))  

def get_nbr_df(metadata_selected, nbr_count, nbr_type, metric):
    nbr_array = metadata.node_name.values
    type_array = metadata.node_type.values
    name_type_array = metadata.node_name_type.values
    metadata_selected_index = metadata_selected.index.values[0]
    if metric == "Manhattan":
        distance_arr = np.load(MANHATTEN_DISTANCE_PATH, mmap_mode="r")         
    elif metric == "Cosine":
        distance_arr = np.load(COSINE_DISTANCE_PATH, mmap_mode="r")    
    nbr_df = pd.DataFrame(list(zip(nbr_array, type_array, name_type_array, distance_arr[metadata_selected_index])), columns=["Concept name", "Concept type", "name_type", "Distance"])
    if nbr_type == DEFAULT_NBR_NODETYPE_SELECTION:
        nbr_df_ = nbr_df.sort_values(by=["Distance", "Concept name"])        
        nbr_df_ = nbr_df_[(nbr_df_["Concept name"]!=metadata_selected["node_name"].values[0]) | (nbr_df_["Concept type"]!=metadata_selected["node_type"].values[0])]
        return nbr_df_.drop_duplicates(subset=["Concept name", "Concept type", "Distance"]).head(nbr_count).reset_index().drop("index", axis=1)        
    else:
        nbr_df_selected = nbr_df[nbr_df["Concept type"]==nbr_type]
        nbr_df_selected_ = nbr_df_selected.sort_values(by=["Distance", "Concept name"])
        nbr_df_selected_ = nbr_df_selected_[(nbr_df_selected_["Concept name"]!=metadata_selected["node_name"].values[0]) | (nbr_df_selected_["Concept type"]!=metadata_selected["node_type"].values[0])]
        return nbr_df_selected_.drop_duplicates(subset=["Concept name", "Concept type", "Distance"]).head(nbr_count).reset_index().drop("index", axis=1)
            
def get_top_features(metadata_selected, feature_metadata, feature_type, feature_count):
    metadata_selected_index = metadata_selected.index.values[0]
    top_feature_index_arr = np.load(TOP_FEATURE_INDEX_PATH, mmap_mode="r")
    feature_metadata_sub = feature_metadata.iloc[top_feature_index_arr[metadata_selected_index]]
    feature_metadata_sub_ = feature_metadata_sub[feature_metadata_sub.node_type==feature_type]
    feature_metadata_sub_.rename(columns={"node_name":"Feature name", "node_type":"Feature type"}, inplace=True)
    feature_metadata_sub_ = feature_metadata_sub_.drop_duplicates(subset=["Feature name", "Feature type"])
    return feature_metadata_sub_[["Feature name", "Feature type"]].head(feature_count).reset_index().drop("index", axis=1)
        

            

            
                        
st.sidebar.header("""
 BEIGE Explorer
""")

metadata = get_metadata()
metadata["color"] = "gray"
metadata["size"] = MARKER_SIZE

dim_sel = get_dim()
color_discrete_map = {"gray":"gray"}
if dim_sel == "3D": 
    fig = px.scatter_3d(metadata, 
                        x="tsne1", y="tsne2", z="tsne3", 
                        color="color",
                        color_discrete_map=color_discrete_map,
                        opacity=0.05, 
                        hover_name="node_name",
                        hover_data=["node_type"]
                       )        
else:
    fig = px.scatter(metadata, 
                     x="tsne1", y="tsne2", 
                     color="color",
                     color_discrete_map=color_discrete_map,
                     opacity=FIG_OPACITY, 
                     hover_name="node_name",
                     hover_data=["node_type"]
                    )

    
node_selected = DEFAULT_SELECTION
node_selected = get_search_term(metadata)
nbr_count = st.sidebar.slider('Neighbor count', MIN_NBR_COUNT, MAX_NBR_COUNT, DEFAULT_NBR_COUNT)
dist_metric = st.sidebar.radio("Distance metric", ["Cosine", "Manhattan"], index=0)
nbr_type = get_nbr_nodetype(metadata)

                
if node_selected != DEFAULT_SELECTION:
    metadata_selected = metadata[metadata.node_name_type==node_selected]    
    metadata_selected["color"] = "darkred"
    color_discrete_map = {"darkred":"darkred"}
    if dim_sel == "3D":
        fig_node_sel = px.scatter_3d(metadata_selected,
                                     x="tsne1", y="tsne2", z="tsne3",
                                     color="color",
                                     color_discrete_map=color_discrete_map,
                                     opacity=1,
                                     hover_name="node_name",
                                     hover_data=["node_type"]
                                    )    
        fig_final = go.Figure(data=fig.data + fig_node_sel.data)        
    else:
        fig_node_sel = px.scatter(metadata_selected,
                                  x="tsne1", y="tsne2",
                                  color="color",
                                  color_discrete_map=color_discrete_map,
                                  opacity=1,
                                  hover_name="node_name",
                                  hover_data=["node_type"]
                                 )
        fig_final = go.Figure(data=fig.data + fig_node_sel.data)     
    feature_explore_status = st.sidebar.checkbox("Explore BEIGE features") 

else:
    fig_final = go.Figure(data=fig.data)
        

if node_selected != DEFAULT_SELECTION:    
    nbr_df = get_nbr_df(metadata_selected, nbr_count, nbr_type, dist_metric)    
    metadata_selected_nbr = metadata[metadata.node_name_type.isin(list(nbr_df.name_type.values))]    
    metadata_selected_nbr["color"] = "orange"
    color_discrete_map = {"orange":"orange"}
    if dim_sel == "3D":
        fig_node_sel_nbr = px.scatter_3d(metadata_selected_nbr,
                                     x="tsne1", y="tsne2", z="tsne3",
                                     color="color",
                                     color_discrete_map=color_discrete_map,
                                     opacity=1,
                                     hover_name="node_name",
                                     hover_data=["node_type"]
                                    )
        fig_final = go.Figure(data=fig.data + fig_node_sel.data + fig_node_sel_nbr.data)
    else:
        fig_node_sel_nbr = px.scatter(metadata_selected_nbr,
                                  x="tsne1", y="tsne2",
                                  color="color",
                                  color_discrete_map=color_discrete_map,
                                  opacity=1,
                                  hover_name="node_name",
                                  hover_data=["node_type"]
                                 )
        fig_final = go.Figure(data=fig.data + fig_node_sel.data + fig_node_sel_nbr.data)
        
        
    
if dim_sel == "3D":
    fig_final.update_layout(
            margin=dict(l=20, r=20, t=20, b=20),   
            width=FIG_WIDTH-200,
            height=FIG_HEIGHT-200,
            showlegend=False,
            scene = dict(
                xaxis = dict(showgrid = False,showticklabels = False,titlefont=dict(size=FONT_SIZE),
                             title="tSNE 1"),
                yaxis = dict(showgrid = False,showticklabels = False,titlefont=dict(size=FONT_SIZE),
                            title="tSNE 2"),
                zaxis = dict(showgrid = False,showticklabels = False,titlefont=dict(size=FONT_SIZE),
                            title="tSNE 3")
            )
           )
else:
    fig_final.update_layout(
            margin=dict(l=20, r=20, t=20, b=20),
            width=FIG_WIDTH+100,
            height=FIG_HEIGHT-300,
            showlegend=False,
            xaxis = dict(showgrid = False,showticklabels = False,titlefont=dict(size=FONT_SIZE),title="tSNE 1"),
            yaxis = dict(showgrid = False,showticklabels = False,titlefont=dict(size=FONT_SIZE),
                        title="tSNE 2")
        )


st.markdown("<h1 style='text-align: center; color: black;'>Biomedical Evidence Integrated Graph Embedding (BEIGE)</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: black;'>Total BEIGE = 7471, Original dimension = 37412</h4>", unsafe_allow_html=True)
st.plotly_chart(fig_final)
if node_selected != DEFAULT_SELECTION:
    st.write("""
        ### Nearest {} {} neighbors in the original space
        """.format(nbr_count, nbr_type))
    st.write(nbr_df[["Concept name", "Concept type", "Distance"]])
    if feature_explore_status:
        feature_metadata = get_feature_metadata()
        feature_count = st.sidebar.slider("Feature count", MIN_FEATURE_COUNT, MAX_FEATURE_COUNT, DEFAULT_FEATURE_COUNT)
        feature_type = st.sidebar.selectbox("Select the feature node type", list(feature_metadata.sort_values(by="node_type").node_type.unique()), index=4)        
        st.write("""
        ### Top {} {} features of BEIGE 
        """.format(feature_count, feature_type)
                )
        st.write(get_top_features(metadata_selected, feature_metadata, feature_type, feature_count))

    
    
    





