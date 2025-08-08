import pandas as pd
from pandas.api.types import CategoricalDtype
from tqdm import tqdm
import os

# Paths and setup
dataframe_paths = [
    "/Users/shayecarver/CellVit/Colorectal_lizard_cells.csv",
    "/Users/shayecarver/CellVit/BRCA_lizard_cells.csv",
    "/Users/shayecarver/CellVit/NSCLC_lizard_cells.csv",
    "/Users/shayecarver/CellVit/Panc_adeno_lizard_cells.csv",
    "/Users/shayecarver/CellVit/Prostate_lizard_cells.csv",
    "/Users/shayecarver/CellVit/RCC_lizard_cells.csv"
]
cancers = ["Colorectal", "BRCA", "NSCLC", "Panc_adeno", "Prostate", "RCC"]

output_path = "/Users/shayecarver/CellVit/lizard_slide_level_features_ALL.csv"
output_path_all = output_path.replace("ALL", "ONLY_all_cells")
output_path_immune = output_path.replace("ALL", "ONLY_immune_cells")

morph_features = [
    'area', 'perimeter', 'eccentricity', 'solidity', 'orientation',
    'major_axis_length', 'minor_axis_length', 'aspect_ratio',
    'circularity', 'centroid_x', 'centroid_y'
]
immune_types = ['Eosinophil', 'Lymphocyte', 'Neutrophil', 'Plasma']
columns = [
    "slide_id", "cell_id", "type_label", "area", "perimeter",
    "eccentricity", "solidity", "orientation", "major_axis_length",
    "minor_axis_length", "aspect_ratio", "circularity", "centroid_x", "centroid_y"
]
chunksize = 1_000_000

def process_and_write_chunk(df, cancer_type, write_header):
    all_cells_df = df.groupby('slide_id')[morph_features].agg('mean').reset_index()
    all_cells_df['type_label'] = 'all_cells'
    all_cells_df['cell_count'] = df.groupby('slide_id').size().values

    immune_subset = df[df['type_label'].isin(immune_types)]
    immune_cells_df = immune_subset.groupby('slide_id')[morph_features].agg('mean').reset_index()
    immune_cells_df['type_label'] = 'immune_cells'
    immune_cells_df['cell_count'] = immune_subset.groupby('slide_id').size().values

    grouped_df = df.groupby(['slide_id', 'type_label']).agg(
        {**{f: 'mean' for f in morph_features}, 'cell_id': 'size'}
    ).reset_index().rename(columns={'cell_id': 'cell_count'})

    all_data = pd.concat([all_cells_df, immune_cells_df, grouped_df], ignore_index=True)
    specific_cell_types = sorted(set(df['type_label']) - set(['all_cells', 'immune_cells']))
    type_order = ['all_cells', 'immune_cells'] + specific_cell_types
    all_data['type_label'] = all_data['type_label'].astype(CategoricalDtype(categories=type_order, ordered=True))
    all_data = all_data.sort_values(by=['slide_id', 'type_label']).reset_index(drop=True)
    all_data['cancer_type'] = cancer_type
    all_cells_map = all_data[all_data['type_label'] == 'all_cells'].set_index('slide_id')['cell_count']
    all_data['prop_type_label'] = all_data['cell_count'] / all_data['slide_id'].map(all_cells_map)
    cols = ['slide_id', 'type_label', 'prop_type_label', 'cell_count', 'cancer_type'] + morph_features
    all_data = all_data[cols]

    all_data.to_csv(output_path, sep='\t', mode='a', header=write_header, index=False)

    all_cells_df = all_data[all_data['type_label'] == 'all_cells'].copy()
    immune_cells_df = all_data[all_data['type_label'] == 'immune_cells'].copy()

    specific_types_df = all_data[
        ~all_data['type_label'].isin(['all_cells', 'immune_cells'])
    ][['slide_id', 'type_label', 'cell_count', 'prop_type_label']]

    count_matrix = specific_types_df.pivot(index='slide_id', columns='type_label', values='cell_count')
    prop_matrix = specific_types_df.pivot(index='slide_id', columns='type_label', values='prop_type_label')
    prop_matrix.columns = [f"prop_{col}" for col in prop_matrix.columns]
    prop_counts = pd.concat([count_matrix, prop_matrix], axis=1)

    all_cells_df = all_cells_df.merge(prop_counts, on='slide_id', how='left')
    immune_cells_df = immune_cells_df.merge(prop_counts, on='slide_id', how='left')

    all_cells_df.to_csv(output_path_all, sep='\t', mode='a', header=write_header, index=False)
    immune_cells_df.to_csv(output_path_immune, sep='\t', mode='a', header=write_header, index=False)

# Main loop
for i, (df_path, cancer_type) in enumerate(tqdm(zip(dataframe_paths, cancers), total=len(cancers))):
    tqdm.write(f"Processing: {cancer_type}")
    reader = pd.read_csv(df_path, header=None, names=columns, chunksize=chunksize)
    buffer = pd.DataFrame()
    write_header = i == 0

    for chunk in tqdm(reader, desc=f"Reading {cancer_type}"):
        chunk = pd.concat([buffer, chunk], ignore_index=True)

        if chunk.empty:
            continue

        last_slide_id = chunk['slide_id'].iloc[-1]
        mask = chunk['slide_id'] == last_slide_id

        complete_chunk = chunk[~mask]
        buffer = chunk[mask]  # always carry forward the last slide_id

        if not complete_chunk.empty:
            process_and_write_chunk(complete_chunk, cancer_type, write_header)
            write_header = False

    # Final chunk (buffer contains complete slide_id if file is sorted)
    if not buffer.empty:
        process_and_write_chunk(buffer, cancer_type, write_header)
