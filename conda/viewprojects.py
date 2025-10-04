import os
import os.path as osp
import pandas as pd
import streamlit as st

def get_proj_csv(name="projects.csv"):
    conda_dir = os.getenv("CONDA_PREFIX")
    assert isinstance(conda_dir, str) and osp.isdir(conda_dir)
    if os.getenv("CONDA_DEFAULT_ENV") != "base":
        conda_dir = osp.abspath(osp.join(conda_dir, "..", ".."))
    return osp.join(conda_dir, name)

# def get_envs_csv(name="conda_envs.csv"):
#     conda_dir = os.getenv("CONDA_PREFIX")
#     assert isinstance(conda_dir, str) and osp.isdir(conda_dir)
#     if os.getenv("CONDA_DEFAULT_ENV") != "base":
#         conda_dir = osp.abspath(osp.join(conda_dir, "..", ".."))
#     return osp.join(conda_dir, name)

st.set_page_config(page_title="Project Environments", layout="wide")
df = pd.read_csv(get_proj_csv())
rows = len(df)
row_height = 35
header_hight = row_height + 5
height = rows*row_height + header_hight
st.title("Projects")
st.dataframe(df, use_container_width=True, height=height)



# def make_project_clickable(row):
#     return f'<a href="{row["github_url"]}">{row["project"]}</a>'
# # display_df['project'] = df.apply(make_project_clickable, axis=1)
# df['project'] = df['project'].apply(make_project_clickable)
# # Convert the DtaFrame to an HTML string
# html_table = df.to_html(escape=False)
# # Display the HTML table in Streamlit
# st.markdown(html_table, unsafe_allow_html=True)