import json
import pandas as pd
import numpy as np
import time
import os
from concept_recognition import *
from entity_linking import *
import math
import warnings
warnings.filterwarnings("ignore")

directory = os.fsencode("./GSC+/Text")
directory_annotations = os.fsencode("./GSC+/Annotations")
output_directory = "./output_concept_recognition/csv_files_GSC_4o"
json_directory = "./output_concept_recognition/json_files_GSC_4o"
with open('list_obsolete.json', 'r') as file:
    json_obsolete_hpo = json.load(file)

encoding = 'to_define_custom'
# reading Text files (iter on 228 GSC+ files)


def extract_phenotypes_info(data, filename_string, json_path, output_file_path):

    print('------------ CONCEPT RECOGNITION -------------')
    start_cr = time.time()
    df_final = concept_recognition_from_text(data, client_open_ai, encoding, json_path, output_file_path)
    end_cr = time.time()

    start_el = time.time()
    if df_final is not None:

        for elem in df_final['phenotype']:

            df_final.loc[df_final['phenotype'] == elem, 'phenotype_extracted_label'] = ''
            df_final.loc[df_final['phenotype'] == elem, 'phenotype_extracted_id'] = ''
            df_final.loc[df_final['phenotype'] == elem, 'phenotype_extracted_label_rag'] = ''
            df_final.loc[df_final['phenotype'] == elem, 'phenotype_extracted_id_rag'] = ''
            df_final.loc[df_final['phenotype'] == elem, 'phenotype_extracted_label_negative'] = ''
            df_final.loc[df_final['phenotype'] == elem, 'phenotype_extracted_id_negative'] = ''
            df_final.loc[df_final['phenotype'] == elem, 'phenotype_extracted_label_rag_negative'] = ''
            df_final.loc[df_final['phenotype'] == elem, 'phenotype_extracted_id_rag_negative'] = ''

            if ((type(elem) == str and elem != 'nan') or ((type(elem) == float and not math.isnan(elem)))):
                print('TERM:', elem)
                print('--------------- ENTITY LINKING --------------')
                top_1, top_1_hpo, top_1_rag, top_1_rag_hpo, matching_keys_top_1, matching_keys_top_1_rag = entity_linking_from_term(elem)
                print(matching_keys_top_1, matching_keys_top_1_rag)
                print('-------------------------------------------')
                df_final.loc[df_final['phenotype'] == elem, 'category'] = str(matching_keys_top_1)
                df_final.loc[df_final['phenotype'] == elem, 'category_rag'] = str(matching_keys_top_1_rag)

                if df_final[df_final['phenotype'] == elem]['negative_phenotype'].values[0] == 0:
                    df_final.loc[df_final['phenotype'] == elem, 'phenotype_extracted_label'] = top_1
                    df_final.loc[df_final['phenotype'] == elem, 'phenotype_extracted_id'] = top_1_hpo
                    df_final.loc[df_final['phenotype'] == elem, 'phenotype_extracted_label_rag'] = top_1_rag
                    df_final.loc[df_final['phenotype'] == elem, 'phenotype_extracted_id_rag'] = top_1_rag_hpo
                    df_final.loc[df_final['phenotype'] == elem, 'phenotype_extracted_label_negative'] = None
                    df_final.loc[df_final['phenotype'] == elem, 'phenotype_extracted_id_negative'] = None
                    df_final.loc[df_final['phenotype'] == elem, 'phenotype_extracted_label_rag_negative'] = None
                    df_final.loc[df_final['phenotype'] == elem, 'phenotype_extracted_id_rag_negative'] = None

                elif df_final[df_final['phenotype'] == elem]['negative_phenotype'].values[0] == 1:
                    df_final.loc[df_final['phenotype'] == elem, 'phenotype_extracted_label_negative'] = top_1
                    df_final.loc[df_final['phenotype'] == elem, 'phenotype_extracted_id_negative'] = top_1_hpo
                    df_final.loc[df_final['phenotype'] == elem, 'phenotype_extracted_label_rag_negative'] = top_1_rag
                    df_final.loc[df_final['phenotype'] == elem, 'phenotype_extracted_id_rag_negative'] = top_1_rag_hpo
                    df_final.loc[df_final['phenotype'] == elem, 'phenotype_extracted_label'] = None
                    df_final.loc[df_final['phenotype'] == elem, 'phenotype_extracted_id'] = None
                    df_final.loc[df_final['phenotype'] == elem, 'phenotype_extracted_label_rag'] = None
                    df_final.loc[df_final['phenotype'] == elem, 'phenotype_extracted_id_rag'] = None


        df_final.to_csv(output_file_path + '.csv', sep=',')

        phenotypes = ''
        phenotypes_unique = ''
        phenotypes_rag = ''
        phenotypes_unique_rag = ''
        phenotypes_negative = ''
        phenotypes_unique_negative = ''
        phenotypes_rag_negative = ''
        phenotypes_unique_rag_negative = ''

        if "phenotype_extracted_id" in df_final:

            phenotypes = df_final["phenotype_extracted_id"].tolist()
            phenotypes_unique = list(set(phenotypes))
            phenotypes_rag = df_final["phenotype_extracted_id_rag"].tolist()
            phenotypes_unique_rag = list(set(phenotypes_rag))
            phenotypes_negative = df_final["phenotype_extracted_id_negative"].tolist()
            phenotypes_unique_negative = list(set(phenotypes_negative))
            phenotypes_rag_negative = df_final["phenotype_extracted_id_rag_negative"].tolist()
            phenotypes_unique_rag_negative = list(set(phenotypes_rag_negative))

        end_el = time.time()

        return pd.Series({
                'ExtractedHPO_OT': phenotypes_unique,
                'ExtractedHPO_OT_RAG': phenotypes_unique_rag,
                'ExtractedHPO_OT_neg': phenotypes_unique_negative,
                'ExtractedHPO_OT_RAG_neg': phenotypes_unique_rag_negative,
                'TimeConceptRecognition': end_cr - start_cr,
                'TimeEntityLinking': end_el - start_el,
                'TotalTime': end_el - start_cr})
    return None

start_time = time.time()

columns_df = ['ID', 'HPO_FINAL', 'HPO_FINAL_OBS', 'TEXT_PAPER', 'TimeConceptRecognition', 'TimeEntityLinking', 'TotalTime', 'ExtractedHPO_OT', 'ExtractedHPO_OT_RAG', 'ExtractedHPO_OT_neg', 'ExtractedHPO_OT_RAG_neg']
df = pd.DataFrame(columns=columns_df)

for filename in os.listdir(directory):

    filename_string = os.fsencode(filename)
    filename_string = filename_string.decode("utf-8")
    json_path = os.path.join(json_directory, '_'.join(['output_json', filename_string]))
    output_file_path = os.path.join(output_directory, '_'.join(['output_csv', filename_string]))

    f = os.path.join(directory, filename)

    with open(f, 'r') as file:
        data = file.read().rstrip()


    hpo_ids = []
    hpo_ids_obs = []
    annotation_file_path = os.path.join(directory_annotations, filename)

    if os.path.exists(annotation_file_path):
        with open(annotation_file_path, 'r', encoding="utf-8") as ann_file:
            for line in ann_file:
                line = line.strip()
                if line:
                    parts = line.split("\t")
                    if len(parts) >= 2:
                        second_column = parts[1]
                        hp_id = second_column.split("|")[0].strip()
                        hp_id = hp_id.replace('_', ':')
                        hpo_ids.append(hp_id)
                        hpo_ids = list(set(hpo_ids))
                        if hp_id in json_obsolete_hpo:
                            hpo_ids_obs.append(hp_id)
                            hpo_ids_obs = list(set(hpo_ids_obs))

    hpo_final_value = ", ".join(hpo_ids) if hpo_ids else None
    hpo_final_value_obs = ", ".join(hpo_ids_obs) if hpo_ids_obs else None

    new_row = {
        'ID': filename_string,
        'TEXT_PAPER': data,
        'HPO_FINAL': hpo_final_value,
        'HPO_FINAL_OBS': hpo_final_value_obs,
        'TimeConceptRecognition': None,
        'TimeEntityLinking': None,
        'TotalTime': None,
        'ExtractedHPO_OT': None,
        'ExtractedHPO_OT_RAG': None,
        'ExtractedHPO_OT_neg': None,
        'ExtractedHPO_OT_RAG_neg': None
    }

    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

try:
    df[['ExtractedHPO_OT', 'ExtractedHPO_OT_RAG', 'ExtractedHPO_OT_neg', 'ExtractedHPO_OT_RAG_neg',
        'TimeConceptRecognition', 'TimeEntityLinking', 'TotalTime']] = df.apply(
        lambda row: extract_phenotypes_info(
            row['TEXT_PAPER'], row['ID'],
            os.path.join(json_directory, '_'.join(['output_json', row['ID']])),
            os.path.join(output_directory, '_'.join(['output_csv', row['ID']]))
        ),
        axis=1
    )
except Exception as e:
    print(e)


df['HPO_FINAL'] = df['HPO_FINAL'].str.split(', ')

for index, row in df.iterrows():
    if row['ExtractedHPO_OT']:
        if None in row['ExtractedHPO_OT']:
            row['ExtractedHPO_OT'].remove(None)
        if "None" in row['ExtractedHPO_OT']:
            row['ExtractedHPO_OT'].remove("None")
        if "nan" in row['ExtractedHPO_OT']:
            row['ExtractedHPO_OT'].remove("nan")
    if row['ExtractedHPO_OT_RAG']:
        if None in row['ExtractedHPO_OT_RAG']:
            row['ExtractedHPO_OT_RAG'].remove(None)
        if "None" in row['ExtractedHPO_OT_RAG']:
            row['ExtractedHPO_OT_RAG'].remove("None")
        if "nan" in row['ExtractedHPO_OT_RAG']:
            row['ExtractedHPO_OT_RAG'].remove("nan")
