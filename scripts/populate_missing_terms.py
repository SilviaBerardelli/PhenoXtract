from optimal_transport import term_missing_syn, term_labels, term_missing_def
import requests
import urllib.parse
import time
import json


filtered_dict = {k: v for k, v in term_labels.items() if k in term_missing_syn}

start = time.time()
terms = filtered_dict.values()
api_key = "{UMLS_API_KEY}"

results_dict = {}
new_dict = {}

for term in terms:
    encoded_term = urllib.parse.quote(term)
    url = f'https://uts-ws.nlm.nih.gov/rest/search/current?string={encoded_term}&apiKey={api_key}'
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        # dict: CUI->Name
        cui_name_dict = {item['ui']: item['name'] for item in data['result']['results']}
        new_dict[term] = list(cui_name_dict.values())
    else:
        print(f"Error for '{term}': Status code {response.status_code}")


file_path = "populated_synonyms_with_umls.json"
with open(file_path, 'w') as file:
    json.dump(new_dict, file, ensure_ascii=False, indent=4)



filtered_dict_def = {k: v for k, v in term_labels.items() if k in term_missing_def}


start = time.time()
terms = filtered_dict_def.values()
results_dict_def = {}
new_dict_def = {}


for term in terms:
    encoded_term = urllib.parse.quote(term)
    search_url = f'https://uts-ws.nlm.nih.gov/rest/search/current?string={encoded_term}&apiKey={api_key}'
    search_response = requests.get(search_url)
    if search_response.status_code == 200:
        search_data = search_response.json()
        term_results = {}
        for item in search_data['result']['results']:
            cui_id = item['ui']
            name = item['name']
            definition_url = f'https://uts-ws.nlm.nih.gov/rest/content/current/CUI/{cui_id}/definitions?apiKey={api_key}'
            definition_response = requests.get(definition_url)
            if definition_response.status_code == 200:
                definition_data = definition_response.json()
                if definition_data['result']:
                    definitions = [def_item['value'] for def_item in definition_data['result'] if def_item['rootSource'] != 'HPO']
                    definition_text = ' '.join(definitions)
                else:
                    definition_text = 'No information available'
            else:
                definition_text = 'Error in definition retrieval'
            term_results[cui_id] = {
                'value': name,
                'definition': definition_text
            }
        results_dict_def[term] = term_results
    else:
        print(f"Error for term '{term}': Status code {search_response.status_code}")

end = time.time()

file_path = "populated_defs_with_umls.json"
with open(file_path, 'w') as file:
    json.dump(results_dict_def, file, ensure_ascii=False, indent=4)


for term, cui_dict in results_dict.items():
    print(f"Term: {term}")
    print("Retrieved information:")
    for cui_id, details in cui_dict.items():
        print(f"CUI ID: {cui_id}")
        print(f"Value: {details['value']}")
        print(f"Definition: {details['definition']}")
        print("-----")
    print("\n")

print("total time:", end - start)


