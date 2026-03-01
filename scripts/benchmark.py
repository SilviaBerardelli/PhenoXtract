import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from concept_recognition import *
from entity_linking import *
import time
import math

output_directory = "output_concept_recognition/"
json_directory = "output_concept_recognition/"
dataset_path = './olida.csv' | './mito.csv' # to define
start_time = time.time()
df = pd.read_csv(dataset_path, sep=",")
df['ExtractedHPO_OT'] = ""

encoding = ''


def extract_phenotypes_info(text, json_path, output_file_path):

    print('------------ CONCEPT RECOGNITION -------------')
    start_cr = time.time()
    df_final = concept_recognition_from_text(text, client_open_ai, encoding, json_path, output_file_path)
    end_cr = time.time()


    start_el = time.time()
    if df_final is not None:

        for elem in df_final['phenotype']:
            if ((type (elem) == str and elem != 'nan') or ((type(elem) == float and not math.isnan(elem)))):
                print('TERM:', elem)
                print('--------------- ENTITY LINKING --------------')
                top_1, top_1_hpo, top_1_rag, top_1_rag_hpo, matching_keys_top_1, matching_keys_top_1_rag = entity_linking_from_term(elem)

                if df_final[df_final['phenotype']==elem]['negative_phenotype'].values[0] == 0:
                    df_final.loc[df_final['phenotype'] == elem, 'phenotype_extracted_label'] = top_1
                    df_final.loc[df_final['phenotype'] == elem, 'phenotype_extracted_id'] = top_1_hpo
                    df_final.loc[df_final['phenotype'] == elem, 'phenotype_extracted_label_rag'] = top_1_rag
                    df_final.loc[df_final['phenotype'] == elem, 'phenotype_extracted_id_rag'] = top_1_rag_hpo
                    df_final.loc[df_final['phenotype'] == elem, 'phenotype_extracted_label_negative'] = None
                    df_final.loc[df_final['phenotype'] == elem, 'phenotype_extracted_id_negative'] = None
                    df_final.loc[df_final['phenotype'] == elem, 'phenotype_extracted_label_rag_negative'] = None
                    df_final.loc[df_final['phenotype'] == elem, 'phenotype_extracted_id_rag_negative'] = None

                elif df_final[df_final['phenotype']==elem]['negative_phenotype'].values[0] == 1:
                    df_final.loc[df_final['phenotype'] == elem, 'phenotype_extracted_label_negative'] = top_1
                    df_final.loc[df_final['phenotype'] == elem, 'phenotype_extracted_id_negative'] = top_1_hpo
                    df_final.loc[df_final['phenotype'] == elem, 'phenotype_extracted_label_rag_negative'] = top_1_rag
                    df_final.loc[df_final['phenotype'] == elem, 'phenotype_extracted_id_rag_negative'] = top_1_rag_hpo
                    df_final.loc[df_final['phenotype'] == elem, 'phenotype_extracted_label'] = None
                    df_final.loc[df_final['phenotype'] == elem, 'phenotype_extracted_id'] = None
                    df_final.loc[df_final['phenotype'] == elem, 'phenotype_extracted_label_rag'] = None
                    df_final.loc[df_final['phenotype'] == elem, 'phenotype_extracted_id_rag'] = None

        df_final.to_csv(output_file_path + '.csv', sep=',')
        #print(df_final)
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
                'TotalTime': end_el - start_cr,
            })
    return None


try:
    df[['ExtractedHPO_OT', 'ExtractedHPO_OT_RAG','ExtractedHPO_OT_neg', 'ExtractedHPO_OT_RAG_neg','TimeConceptRecognition', 'TimeEntityLinking', 'TotalTime']] = df.apply(
        lambda row: extract_phenotypes_info(
            row['TEXT_PAPER'],
            os.path.join(json_directory, '_'.join(['output_json', row['ID']])),
            os.path.join(output_directory, '_'.join(['output_csv', row['ID']]))
        ),
        axis=1
    )
except Exception as e:
    print(e)


df['HPO_FINAL'] = df['HPO_FINAL'].str.split(',')

for index, row in df.iterrows():
    if row['ExtractedHPO_OT']:
        if None in row['ExtractedHPO_OT']:
            row['ExtractedHPO_OT'].remove(None)
        if "nan" in row['ExtractedHPO_OT']:
            row['ExtractedHPO_OT'].remove("nan")
    if row['ExtractedHPO_OT_RAG']:
        if None in row['ExtractedHPO_OT_RAG']:
            row['ExtractedHPO_OT_RAG'].remove(None)
        if "nan" in row['ExtractedHPO_OT_RAG']:
            row['ExtractedHPO_OT_RAG'].remove("nan")